import random
from typing import List, Optional, Union

import gym
import onnx
import onnxruntime as ort
from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torchvision
from models.world_model import WorldModelOutput


class Encoder(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    def forward(self, observations):
        return self.tokenizer.encode(observations, should_preprocess=True).tokens

class Decoder(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    def forward(self, tokens):
        z = self.tokenizer.embedding(tokens)
        z = rearrange(z, 'b (h w) e -> b e h w', h=int(np.sqrt(4*4)))
        return self.tokenizer.decode(z, should_postprocess=True)


class WorldModelEnv:
    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device], env: Optional[gym.Env] = None) -> None:
        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()
        self.keys_values_wm, self.obs_tokens, self._num_observations_tokens = None, None, None
        self.env = env

        # Export world model
        dummy_tokens = torch.zeros(4, 16, dtype=torch.long)
        dummy_kv = world_model.transformer.generate_empty_keys_values(4, self.world_model.config.max_tokens // 2)
        dummy_kv._size = self.world_model.config.max_tokens // 2
        torch.onnx.export(self.world_model, (dummy_tokens, dummy_kv.get()), '/tmp/wm.onnx', opset_version=11,
          input_names=['tokens', 'past'],
          output_names=['output_seq', 'logits_obs', 'logits_reward', 'logits_done', 'kv'],
          dynamic_axes={
            'tokens': {0: 'b', 1: 't'},
            'past': {2: 'b', 4: 't_past'},
            'output_seq': {0: 'b', 1: 't', 2: 'e'},
            'logits_obs': {0: 'b', 1: 't_obs'},
            'logits_reward': {0: 'b', 1: 't_reward'},
            'logits_done': {0: 'b', 1: 't_done'},
            'kv': {2: 'b', 3: 'nh', 4: 't', 5: 'hs'}})

        # Export encoder
        dummy_img = torch.zeros(4, 3, 64, 64)
        encoder = Encoder(self.tokenizer)
        torch.onnx.export(encoder, dummy_img, '/tmp/enc.onnx', opset_version=11,
          input_names=['img'],
          output_names=['tokens'],
          dynamic_axes={
            'img': {0: 'b', 1: 't'},
            'tokens': {0: 'b', 1: 't'}})

        # Export decoder
        dummy_tokens = torch.zeros(8, 4*4).long()
        decoder = Decoder(self.tokenizer)
        torch.onnx.export(decoder, dummy_tokens, '/tmp/dec.onnx', opset_version=11,
          input_names=['tokens'],
          output_names=['img'],
          dynamic_axes={
            'tokens': {0: 'b', 1: 't'},
            'img': {0: 'b', 1: 't'}})

        # Runners
        self.wm_runner = ort.InferenceSession('/tmp/wm.onnx', None, ['CPUExecutionProvider'])
        self.enc_runner = ort.InferenceSession('/tmp/enc.onnx', None, ['CPUExecutionProvider'])
        self.dec_runner = ort.InferenceSession('/tmp/dec.onnx', None, ['CPUExecutionProvider'])


    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        obs = torchvision.transforms.functional.to_tensor(self.env.reset()).to(self.device).unsqueeze(0)  # (1, C, H, W) in [0., 1.]
        return self.reset_from_initial_observations(obs)

    @torch.no_grad()
    def reset_from_initial_observations(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        obs_tokens = self.enc_runner.run(None, {'img': observations.numpy()})[0]  # (B, C, H, W) -> (B, K)
        _, num_observations_tokens = obs_tokens.shape
        if self.num_observations_tokens is None:
            self._num_observations_tokens = num_observations_tokens

        obs_tokens = torch.as_tensor(obs_tokens)
        self.refresh_keys_values_with_initial_obs_tokens(obs_tokens)
        self.obs_tokens = obs_tokens

        return self.decode_obs_tokens()

    @torch.no_grad()
    def refresh_keys_values_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:
        n, num_observations_tokens = obs_tokens.shape
        assert num_observations_tokens == self.num_observations_tokens
        self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=self.world_model.config.max_tokens)
        outputs_wm = self.wm_runner.run(None, {'tokens': obs_tokens.numpy(), 'past': self.keys_values_wm.get().numpy()})
        outputs_wm = WorldModelOutput(*outputs_wm)
        self.keys_values_wm.update(torch.as_tensor(outputs_wm.keys_values))

    @torch.no_grad()
    def step(self, action: Union[int, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True) -> None:
        assert self.keys_values_wm is not None and self.num_observations_tokens is not None

        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1
        if self.keys_values_wm.size + num_passes > self.world_model.config.max_tokens:
            self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)

        token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        token = token.reshape(-1, 1).to(self.device)  # (B, 1)

        output_sequence, obs_tokens = [], []
        for k in range(num_passes):  # assumption that there is only one action token.
            outputs_wm = self.wm_runner.run(None, {'tokens': token.numpy(), 'past': self.keys_values_wm.get().numpy()})
            outputs_wm = [torch.as_tensor(x) for x in outputs_wm]
            outputs_wm = WorldModelOutput(*outputs_wm)
            output_sequence.append(outputs_wm.output_sequence)
            self.keys_values_wm.update(outputs_wm.keys_values)

            if k == 0:
                reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
                done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)

            if k < self.num_observations_tokens:
                token = Categorical(logits=outputs_wm.logits_observations).sample()
                obs_tokens.append(token)

        output_sequence = torch.cat(output_sequence, dim=1)   # (B, 1 + K, E)
        self.obs_tokens = torch.cat(obs_tokens, dim=1)        # (B, K)

        obs = self.decode_obs_tokens() if should_predict_next_obs else None
        return obs, reward, done, None

    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self) -> List[Image.Image]:
        rec = self.dec_runner.run(None, {'tokens': self.obs_tokens.numpy()})[0]
        rec = torch.as_tensor(rec)
        return torch.clamp(rec, 0, 1)

    @torch.no_grad()
    def render(self):
        assert self.obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]
