import os
import sys
import time
import torch
from models.transformer import Transformer, TransformerConfig

torch.manual_seed(42)

config = TransformerConfig(
    tokens_per_block = 17,
    max_blocks = 20,
    attention = 'causal',
    num_layers = 10,
    num_heads = 4,
    embed_dim = 256,
    embed_pdrop = 0.1,
    resid_pdrop = 0.1,
    attn_pdrop = 0.1)

bs = 4
transformer = Transformer(config).eval()
data = torch.rand(bs, 32, 256)
with torch.no_grad():
  result, _ = transformer(data) # run once to load everything in
assert data.shape == result.shape

ref_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'transformer_ref.pkl')
if '--update'  in sys.argv:
  torch.save({'data': data, 'result': result}, ref_path)
  print("Updated refs")
else:
  ref = torch.load(ref_path)
  torch.testing.assert_close(ref['data'], data)
  torch.testing.assert_close(ref['result'], result)
  print("Refs match")

# Test with kv cache
kv_cache = transformer.generate_empty_keys_values(n=bs, max_tokens=32)
for i in range(32):
  with torch.no_grad():
    step, kv = transformer(data[:,i:i+1], kv_cache)
  torch.testing.assert_close(step, result[:,i:i+1])
  for kv_layer, kvl in zip(kv_cache, kv):
    kv_layer.update(kvl)

# Test timing w/cache
print("Timing, w/o cache, t=32")
for _ in range(10):
  st = time.monotonic()
  with torch.no_grad():
    transformer(data)
  tt = time.monotonic() - st
  print(f"- Performed inference in {tt*1000:.2f}ms")
  assert tt < .02

# Test timing w/cache
print("Timing, w/cache, t=1")
kv_cache = transformer.generate_empty_keys_values(n=bs, max_tokens=10)
for i in range(10):
  st = time.monotonic()
  with torch.no_grad():
    _, kv = transformer(data[:,i:i+1], kv_cache)
  for kv_layer, kvl in zip(kv_cache, kv):
    kv_layer.update(kvl)
  tt = time.monotonic() - st
  print(f"Performed inference in {tt*1000:.2f}ms")
  assert tt < .02
