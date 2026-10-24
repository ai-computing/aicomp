"""Compile an isolated diagnostic artifact using SDK output_hidden_states support."""
from pathlib import Path
import json,time
import optimum.rbln as rbln
root=Path(__file__).resolve().parent
artifact=root/'atom270-artifacts-allcoreprobe'/'Qwen--Qwen3-4B-tp1-s9216-b4-k5'
assert not artifact.exists(), 'Preserve existing diagnostic artifacts'
config=dict(batch_size=4,max_seq_len=9216,tensor_parallel_size=1,kvcache_num_blocks=5,kvcache_block_size=9216,output_hidden_states=True,create_runtimes=False)
from atom270_allcore_probe import CoreProbeWrapper
rbln.RBLNQwen3ForCausalLM._decoder_wrapper_cls=CoreProbeWrapper
start=time.perf_counter()
print('HIDDEN_COMPILE_START '+json.dumps(config),flush=True)
model=rbln.RBLNQwen3ForCausalLM.from_pretrained('Qwen/Qwen3-4B',export=True,rbln_config=config)
model.save_pretrained(artifact)
print('HIDDEN_COMPILE_COMPLETE '+json.dumps(dict(seconds=time.perf_counter()-start,config=config)),flush=True)
