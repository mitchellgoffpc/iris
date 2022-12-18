from typing import Tuple

import numpy as np
import torch


class KVCache:
    def __init__(self, num_samples: int, num_heads: int, max_tokens: int, embed_dim: int, device: torch.device) -> None:
        assert embed_dim % num_heads == 0
        self._n, self._cache, self._size = num_samples, None, None
        self._cache = torch.empty(2, num_samples, num_heads, max_tokens, embed_dim // num_heads, device=device)  # (B, nh, T, hs)
        self._size = 0

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        _, n, num_heads, _, head_dim = self._cache.shape
        return n, num_heads, self._size, head_dim

    def reset(self) -> None:
        self._cache[:] = 0
        self._size = 0

    def get(self) -> torch.Tensor:
        return self._cache[:, :, :, :self._size, :]

    def update(self, k: torch.Tensor, v: torch.Tensor) -> None:
        assert k.size(2) == v.size(2)
        for x in (k, v):
          assert x.ndim == self._cache.ndim-1
          assert all([x.size(i) == self._cache.size(i+1) for i in (0, 1, 3)])
          assert self._size + x.size(2) <= self._cache.size(3)
        self._cache[:, :, :, self._size : self._size + k.size(2)] = torch.stack([k, v], 0)
        self._size += k.size(2)


class KeysValues:
    def __init__(self, n: int, num_heads: int, max_tokens: int, embed_dim: int, num_layers: int, device: torch.device) -> None:
        self._keys_values = tuple([KVCache(n, num_heads, max_tokens, embed_dim, device) for _ in range(num_layers)])

    def __getitem__(self, key: int) -> KVCache:
        return self._keys_values[key]

    def __len__(self):
        return len(self._keys_values)

    @property
    def size(self):
        return self._keys_values[0].shape[2]

    def reset(self) -> None:
        for kv_cache in self._keys_values:
            kv_cache.reset()
