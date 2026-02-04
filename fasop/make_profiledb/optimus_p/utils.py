from datetime import datetime
from collections import defaultdict

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(line: str) -> None:
    print(f"[{ts()}] {line}", flush=True)

def log_special_nodes(rank: int, world_size: int, special_nodes: dict,
                      *, min_span: int = 2, topk: int = 5):
    s = _summerize_special_nodes(special_nodes)
    if rank == 0:
        log(f" special_nodes by rank | world_size={world_size} | far=span>={min_span} | topk={topk}")
    log(f" special_nodes | total={s['total']} far={s['far']} max_span={s['max_span']} dropped_submod_adj={s['dropped_submod_adj']}")
    if s["top_far"]:
        log(f"      top_far: " + ", ".join(s["top_far"]))

def _summerize_special_nodes(special_nodes: dict, *, min_span: int = 2, topk: int = 5):
    if not special_nodes:
        return dict(total=0, far=0, max_span=0, dropped_submod_adj=0, top_far=[])
    total = len(special_nodes)

    # submod_k(k,k+1) -> dropped
    dropped_submod_adj = 0
    filtered = {}
    for name, (src, dst) in special_nodes.items():
        if str(name).startswith("submod_") and (dst - src) == 1:
            dropped_submod_adj += 1
            continue
        filtered[name] = (src, dst)

    # Extract far special nodes
    far = []
    for name, (src, dst) in filtered.items():
        span = dst - src
        if span >= min_span:
            far.append((span, name, src, dst))

    far.sort(reverse=True)
    max_span = far[0][0] if far else 0
    top_far = [f"{name}: stage{src}→stage{dst} (span={span})" for span, name, src, dst in far[:topk]]      

    return dict(total=total, far=len(far), max_span=max_span, dropped_submod_adj=dropped_submod_adj, top_far=top_far)