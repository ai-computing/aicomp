"""Measured kernel overlap of this Qwen3 mixed128/512/1024 direct-KV H2D trace."""
import argparse
import bisect
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

# Four BF16 KV slots, each [prompt_tokens-1,8,128]. Excludes model/input transfers.
KV_BYTES=(1040384,4186112,8380416)


def summarize(db):
    kernels=defaultdict(list)
    for start,end,device,context,pid in db.execute(
            'select start,end,deviceId,contextId,globalPid from CUPTI_ACTIVITY_KIND_KERNEL order by start'):
        intervals=kernels[device,context,pid]
        if intervals and start<=intervals[-1][1]:
            intervals[-1]=(intervals[-1][0],max(end,intervals[-1][1]))
        else:intervals.append((start,end))
    ends={key:[end for _,end in intervals] for key,intervals in kernels.items()}
    result=dict(kv_copies=0,copy_duration_ns=0,kernel_overlap_ns=0,copies_with_kernel_overlap=0,bytes=0)
    sizes={}
    for start,end,device,context,pid,nbytes in db.execute(
            'select start,end,deviceId,contextId,globalPid,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 order by start'):
        if nbytes not in KV_BYTES:continue
        intervals=kernels.get((device,context,pid),[])
        index=bisect.bisect_right(ends.get((device,context,pid),[]),start)
        overlap=0
        for i in range(index,len(intervals)):
            a,b=intervals[i]
            if a>=end:break
            overlap+=max(0,min(end,b)-max(start,a))
        for row in (result,sizes.setdefault(str(nbytes),dict(kv_copies=0,copy_duration_ns=0,kernel_overlap_ns=0,copies_with_kernel_overlap=0,bytes=0))):
            row['kv_copies']+=1;row['bytes']+=nbytes
            row['copy_duration_ns']+=end-start;row['kernel_overlap_ns']+=overlap
            row['copies_with_kernel_overlap']+=int(overlap>0)
    for row in [result,*sizes.values()]:
        row['overlap_fraction']=row['kernel_overlap_ns']/row['copy_duration_ns'] if row['copy_duration_ns'] else None
    result['by_copy_bytes']=sizes
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite');parser.add_argument('--out',required=True)
    args=parser.parse_args()
    with sqlite3.connect(Path(args.sqlite).resolve().as_uri()+'?mode=ro',uri=True) as db:
        result=summarize(db)
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
