import json
from pathlib import Path
root=Path(__file__).resolve().parent

def union(intervals):
 out=[]
 for a,b in sorted(intervals):
  if out and a <= out[-1][1]:out[-1][1]=max(out[-1][1],b)
  else:out.append([a,b])
 return out

def intersection(a,b):
 i=j=0; total=0
 while i<len(a) and j<len(b):
  total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
  if a[i][1] < b[j][1]:i+=1
  else:j+=1
 return total

results=[]
for flag in [0,1]:
 d=json.loads((root/f'chunk47-qwen06-trace-{flag}.json').read_text())
 events=d['traceEvents']
 kernels=[e for e in events if e.get('cat')=='kernel' and e['args'].get('device')==0]
 gemms=[e for e in kernels if 'gemm' in e['name'].lower()]
 copies=[e for e in events if e.get('cat')=='gpu_memcpy' and e['args'].get('device')==0 and 'DtoH' in e['name'] and e['args'].get('bytes',0)>1024*1024]
 ki=union([(e['ts'],e['ts']+e['dur']) for e in kernels]);gi=union([(e['ts'],e['ts']+e['dur']) for e in gemms]);ci=union([(e['ts'],e['ts']+e['dur']) for e in copies])
 results.append(dict(copy_overlap=bool(flag),dtoh_count=len(copies),dtoh_bytes=sum(e['args']['bytes'] for e in copies),dtoh_union_ms=sum(b-a for a,b in ci)/1000,kernel_intersection_ms=intersection(ci,ki)/1000,gemm_intersection_ms=intersection(ci,gi)/1000,gemm_count=len(gemms),copy_streams=sorted(set(e['args']['stream'] for e in copies)),kernel_streams=sorted(set(e['args']['stream'] for e in kernels))))
 del events,d,kernels,gemms,copies
out={'method':'Intersection of unioned device0 CUDA kernel/GEMM intervals and >1MiB device0 DtoH intervals. Profiler runs separate from unprofiled latency runs; not critical-path attribution.', 'rows':results}
(root/'chunk47-qwen06-trace-summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
