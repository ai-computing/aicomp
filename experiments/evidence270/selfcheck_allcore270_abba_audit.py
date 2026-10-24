"""Historical raw failure must stay failed; inconsistent flags/timing must be rejected."""
import copy,json
from pathlib import Path
from audit_atom270_allcore_abba import audit_arm
A=Path(__file__).resolve().parent
def subset(name):
 d=json.loads((A/name).read_text());d['prompts']={k:v for k,v in d['prompts'].items() if k in ['512','8192']};d['references']={k:v for k,v in d['references'].items() if k in ['512:128','8192:128']}
 d['warmups']=[r for r in d['warmups'] if '-'.join(r['id'].split('-')[-2:]) in ['512-128','8192-128']]
 d['correctness_batches']=[b for b in d['correctness_batches'] if b['input_tokens'] in [512,8192] and b['output_tokens']==128]
 d['batches']=[b for b in d['batches'] if b['input_tokens'] in [512,8192] and b['output_tokens_per_request']==128]
 return d
bad=subset('atom270-qwen4b-tp1-b4-native.json');assert not audit_arm(bad)['parity_passed']
good=subset('atom270-qwen4b-allcore-http.json');assert audit_arm(good)['parity_passed']
for field in ['flag','time','token','missing']:
 d=copy.deepcopy(good)
 if field=='flag':d['correctness_batches'][0]['rows'][0]['tokens_equal']=False
 elif field=='time':d['batches'][0]['summary']['output_tok_s']*=2
 elif field=='token':d['correctness_batches'][0]['rows'][0]['tokens'][0]+=1
 else:d['batches'].pop()
 try:audit_arm(d)
 except AssertionError:pass
 else:raise AssertionError('Bad evidence accepted:'+field)
print('PASS: historical parity failure preserved; valid candidate accepted;4 corrupted cases rejected')
