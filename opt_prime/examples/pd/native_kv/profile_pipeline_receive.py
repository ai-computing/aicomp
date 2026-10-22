"""Diagnostic launcher: record receive intervals; optionally change GIL switch interval."""
import sys,time,json,runpy,os
from pathlib import Path

import pinned_receiver
original=pinned_receiver.PinnedAssembler
out=open(os.environ['RECEIVE_PROFILE'],'w')
class Timed(original):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.events=[]
 def receive(self,h,s):
  t=time.perf_counter();c=time.thread_time();done=super().receive(h,s)
  self.events.append([t,time.perf_counter(),time.thread_time()-c])
  if done:
   out.write(json.dumps({'positions':self.positions,'events':self.events})+'\n');out.flush()
  return done
pinned_receiver.PinnedAssembler=Timed
sys.setswitchinterval(float(os.environ.get('SWITCH_INTERVAL','0.005')))
runpy.run_path(str(Path(__file__).with_name('serve_gpu.py')),run_name='__main__')
