"""Audit transport parity separately from full compiled-batch occupancy."""
import hashlib,json
from pathlib import Path
from audit_http265 import audit_result
A=Path(__file__).resolve().parent

def audit_mode(data,log,mode):
 result={'responses':audit_result(data,'atom',mode,expected_slots=4,expected_atom_devices=1)}
 assert data['kv_blocks']==5
 def one(marker):
  rows=[json.loads(l[len(marker):]) for l in log.splitlines() if l.startswith(marker)];assert len(rows)==1,marker
  return rows[0]
 geometry=one('ATOM270_COMPILED ');assert geometry==dict(batch_size=4,kv_blocks=5,context=9216,tensor_parallel_size=1)
 binding=one('ATOM270_BINDING ');mapping=binding['physical_to_visible']
 assert binding['visible_count']==1 and len(mapping)==1 and set(mapping.values())=={0}
 assert all(k.isdigit() for k in mapping) and set(binding['opened'])=={f'/dev/rbln{k}' for k in mapping}
 hist=one('ATOM270_ACTIVE_LANES ');assert len(hist)==5 and all(type(v) is int and v>=0 for v in hist) and sum(hist)>0
 result.update(geometry=geometry,binding=binding,active_lane_histogram=hist,full_batch_observed=hist[4]>0)
 return result

def main():
 outer=json.loads((A/'atom270-allcore-pd-outer.json').read_text());assert outer['state']=='complete' and outer['exit_code']==outer['cleanup_exit_code']==0
 queue=json.loads((A/'atom270-allcore-pd-arms.json').read_text());assert [x['mode'] for x in queue]==['whole','stream'] and all(x['exit_code']==0 for x in queue)
 report={'modes':{},'limits':'Within-route HTTP parity and whole/stream parity; occupancy reported independently. Single timed batches, not a matched native/PD speedup or general quality result.'};data={}
 for mode in ['whole','stream']:
  p=A/f'atom270-qwen4b-allcore-pd-{mode}.json';d=json.loads(p.read_text());data[mode]=d
  log=(A/f'atom270-qwen4b-allcore-pd-{mode}.server0.log').read_text()
  report['modes'][mode]=audit_mode(d,log,mode);report['modes'][mode]['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 assert report['modes']['whole']['binding']==report['modes']['stream']['binding']
 assert data['whole']['prompts']==data['stream']['prompts']
 matches={k:v['tokens']==data['stream']['references'][k]['tokens'] for k,v in data['whole']['references'].items()}
 assert all(matches.values());report['whole_stream_references_equal']=matches
 native=json.loads((A/'atom270-qwen4b-allcore-http.json').read_text());assert native['prompts']==data['whole']['prompts']
 report['historical_native_reference_matches']={k:v['tokens']==native['references'][k]['tokens'] for k,v in data['whole']['references'].items()}
 report['full_batch_observed_both_modes']=all(x['full_batch_observed'] for x in report['modes'].values())
 report['timings']={mode:[{'input':b['input_tokens'],'output_per_request':b['output_tokens_per_request'],'summary':b['summary']} for b in d['batches']] for mode,d in data.items()}
 (A/'atom270-allcore-pd-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
