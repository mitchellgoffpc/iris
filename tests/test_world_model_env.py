import os
import sys
import time
import torch

from envs.world_model_env import WorldModelEnv
from models.transformer import TransformerConfig
from models.world_model import WorldModel, WorldModelOutput
from models.tokenizer import Tokenizer, Encoder, Decoder, EncoderDecoderConfig

torch.manual_seed(42)

transformer_config = TransformerConfig(
    tokens_per_block = 17,
    max_blocks = 20,
    attention = 'causal',
    num_layers = 10,
    num_heads = 4,
    embed_dim = 256,
    embed_pdrop = 0.1,
    resid_pdrop = 0.1,
    attn_pdrop = 0.1)

tokenizer_config = EncoderDecoderConfig(
  resolution = 64,
  in_channels = 3,
  z_channels = 512,
  ch = 64,
  ch_mult = [1, 1, 1, 1, 1],
  num_res_blocks = 2,
  attn_resolutions = [8, 16],
  out_ch = 3,
  dropout = 0.0)

bs = 4
tokenizer = Tokenizer(512, 512, Encoder(tokenizer_config), Decoder(tokenizer_config)).eval()
world_model = WorldModel(512, 512, transformer_config).eval()
env = WorldModelEnv(tokenizer, world_model, 'cpu')
initial_obs = torch.rand(bs, 3, 64, 64)
env.reset_from_initial_observations(initial_obs)

obs, rewards, dones = [], [], []
for _ in range(10):
  st = time.monotonic()
  o,r,d,_ = env.step(action=torch.tensor([0]*bs).long(), should_predict_next_obs=True)
  obs.append(o)
  rewards.append(r)
  dones.append(d)
  tt = time.monotonic() - st
  print(f"- Performed step in {tt*1000:.2f}ms")

obs = torch.stack(obs)
ref_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'refs/world_model_env_ref.pkl')
if '--update'  in sys.argv:
  torch.save({'initial_obs': initial_obs, 'obs': obs, 'rewards': rewards, 'dones': dones}, ref_path)
  print("Updated refs")
else:
  ref = torch.load(ref_path)
  torch.testing.assert_close(ref['initial_obs'], initial_obs)
  torch.testing.assert_close(ref['obs'], obs)
  torch.testing.assert_close(ref['rewards'], rewards)
  torch.testing.assert_close(ref['dones'], dones)
  print("Refs match")
