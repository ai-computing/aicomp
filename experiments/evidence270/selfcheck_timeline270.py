from concurrent.futures import Future
from types import SimpleNamespace as NS
from timeline270 import TimelineProbe
class Clock:
    t=0.
    def __call__(self): return self.t
clock=Clock(); f=Future(); waiting=Future()
b=NS(decoding=[1,2],prefilling=[NS(pending=waiting)],waiting=[3,4],_decode_step=lambda:'ok',worker=NS(submit=lambda *a,**k:f))
p=TimelineProbe(clock=clock);p.install(b)
r=NS(input_ids=[1,2,3]);assert b.worker.submit(r,slot=2) is f
clock.t=2;f.set_result('done');clock.t=3
assert b._decode_step()=='ok'
s=p.snapshot();assert s['remote']==[dict(id=0,slot=2,input_tokens=3,start=0.,end=2.,error=False)]
assert s['decode']==[dict(start=3.,end=3.,lanes=2,prefilling=1,ready=0,waiting=2,error=False)]
waiting.set_result(None);clock.t=4; b._decode_step();assert p.snapshot()['decode'][-1]['ready']==1
s['decode'][0]['lanes']=99;assert p.snapshot()['decode'][0]['lanes']==2
boom=RuntimeError('decode')
def fail():raise boom
b2=NS(decoding=[1],prefilling=[],waiting=[],_decode_step=fail,worker=None)
p2=TimelineProbe(clock=clock);p2.install(b2)
try:b2._decode_step()
except RuntimeError as e: assert e is boom
else:raise AssertionError('exception hidden')
assert p2.snapshot()['decode'][0]['error']
print('PASS timeline timestamps, queue state, Future identity, snapshot isolation and error preservation')
