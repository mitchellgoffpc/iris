from typing import Tuple

import numpy as np
import torch


class KeysValues:
    def __init__(self, num_samples: int, num_heads: int, max_tokens: int, embed_dim: int, num_layers: int, device: torch.device) -> None:
        assert embed_dim % num_heads == 0
        self._n, self._cache, self._size = num_samples, None, None
        self._cache = torch.empty(num_layers, 2, num_samples, num_heads, max_tokens, embed_dim // num_heads, device=device)  # (nl, 2, B, nh, T, hs)
        self._size = 0

    @property
    def size(self) -> int:
      return self._size

    def reset(self) -> None:
        self._cache[:] = 0
        self._size = 0

    def get(self) -> torch.Tensor:
        return self._cache[:, :, :, :, :self._size, :]

    def update(self, kv: torch.Tensor) -> None:
        assert kv.ndim == self._cache.ndim
        assert all([kv.size(i) == self._cache.size(i) for i in (0, 1, 2, 3, 5)])
        assert self._size + kv.size(4) <= self._cache.size(4)
        self._cache[:, :, :, :, self._size : self._size + kv.size(4)] = kv
        self._size += kv.size(4)
