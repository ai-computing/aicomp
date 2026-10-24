"""Read-only process/cgroup/listener snapshots for HTTP failure diagnosis."""
from pathlib import Path
import time

def snapshot(procs):
 out={'monotonic_s':time.monotonic(),'servers':[],'cgroup':{}}
 for proc in procs:
  row={'pid':proc.pid,'returncode':proc.poll()}
  try:
   row['status']=[s for s in Path(f'/proc/{proc.pid}/status').read_text().splitlines() if s.startswith(('Name:','State:','VmRSS:','VmHWM:','Threads:'))]
  except OSError as e:row['status_error']=type(e).__name__
  out['servers'].append(row)
 for name in ['memory.events','memory.current','memory.max','memory.peak']:
  try:out['cgroup'][name]=Path('/sys/fs/cgroup',name).read_text().strip()
  except OSError as e:out['cgroup'][name]=type(e).__name__
 try:
  lines=Path('/proc/net/netstat').read_text().splitlines()
  for a,b in zip(lines[::2],lines[1::2]):
   if a.startswith('TcpExt:'):
    counters=dict(zip(a.split()[1:],b.split()[1:]));out['listen']={k:int(counters[k]) for k in ['ListenOverflows','ListenDrops']};break
 except (OSError,KeyError,ValueError) as e:out['listen_error']=type(e).__name__
 return out
