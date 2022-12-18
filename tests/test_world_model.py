import os
import sys
import time
import torch
from models.transformer import TransformerConfig
from models.world_model import WorldModel, WorldModelOutput

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
world_model = WorldModel(512, 512, config).eval()
data = torch.randint(512, size=(bs, 32))
with torch.no_grad():
  result = world_model(data) # run once to load everything in

ref_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'world_model_ref.pkl')
if '--update'  in sys.argv:
  torch.save({'data': data, 'result': result}, ref_path)
  print("Updated refs")
else:
  ref = torch.load(ref_path)
  torch.testing.assert_close(ref['data'], data)
  torch.testing.assert_close(ref['result'], result)
  print("Refs match")

# Test with kv cache
kv_cache = world_model.transformer.generate_empty_keys_values(n=bs, max_tokens=32)
idxs = {k:0 for k in WorldModelOutput._fields}
for i in range(32):
  with torch.no_grad():
    step = world_model(data[:,i:i+1], kv_cache)
  for k in idxs:
    s, r, j = step._asdict()[k], result._asdict()[k], idxs[k]
    torch.testing.assert_close(s, r[:,j:j+s.shape[1]])
    idxs[k] += s.shape[1]


# Test timing w/cache
print("Timing, w/o cache, t=32")
for _ in range(10):
  st = time.monotonic()
  with torch.no_grad():
    world_model(data)
  tt = time.monotonic() - st
  print(f"- Performed inference in {tt*1000:.2f}ms")
  assert tt < .02

# Test timing w/cache
print("Timing, w/cache, t=1")
kv_cache = world_model.transformer.generate_empty_keys_values(n=bs, max_tokens=10)
for i in range(10):
  st = time.monotonic()
  with torch.no_grad():
    _ = world_model(data[:,i:i+1], kv_cache)
  tt = time.monotonic() - st
  print(f"Performed inference in {tt*1000:.2f}ms")
  assert tt < .01
