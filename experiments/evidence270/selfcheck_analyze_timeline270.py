from analyze_timeline270 import summarize
trace={'remote':[dict(id=0,slot=0,input_tokens=128,start=0.,end=2.,error=False),dict(id=1,slot=1,input_tokens=128,start=0.,end=5.,error=False)],'decode':[dict(start=2.1,end=2.2,lanes=1,prefilling=1,ready=0,waiting=14,error=False),dict(start=5.1,end=5.3,lanes=2,prefilling=0,ready=0,waiting=14,error=False)]}
s=summarize(trace)
assert s['remote']['completed']==2 and s['remote']['completion_gap_s']['median']==3.
assert s['remote']['submit_to_completion_s']['median']==3.5
assert s['decode']['lane_histogram']=={'1':1,'2':1}
assert s['decode']['calls_with_ready_prefill']==0 and s['decode']['calls_with_waiting']==2
assert s['decode']['max_lanes']==2
assert abs(s['decode']['host_call_s']['sum']-.3)<1e-8
try:summarize({'remote':[], 'decode':[dict(trace['decode'][0],end=1.)]})
except ValueError:pass
else:raise AssertionError('invalid interval accepted')
s2=summarize({'remote':[], 'decode':[]});assert s2['decode']['host_call_s']['median'] is None
print('PASS interval summaries, queue-state counts, empty data and malformed interval rejection')

assert s['remote']['backlogged_completion_gap_s']['median']==3.
idle={'remote':[trace['remote'][0],dict(trace['remote'][1],start=4.)], 'decode':[]}
assert summarize(idle)['remote']['backlogged_completion_gap_s']['count']==0
print('PASS completion gap excludes requests submitted after previous completion')
