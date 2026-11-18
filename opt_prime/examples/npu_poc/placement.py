"""Placement — maps the (PP, TP) grid onto physical cards.  Integration items #2·#4.

This is where the two axes go to different owners:
    PP is split by the PoC core — one device group per stage
    TP is split by the backend — the compiler shards within that group

On ATOM+, TP lives inside the rebel binary (collectives compiled in).
opt_prime's own TP (prepare_tp_group → ColwiseParallel/RowwiseParallel + DTensor)
assumes GPU·NCCL and is not used — stacking both TPs would shard twice.
"""

from __future__ import annotations

import subprocess


def assign(n_stages: int, tp_size: int, device_ids: list[int] | None,
           caps) -> list[list[int]]:
    """Return, for each stage, the list of devices that stage uses."""
    bad = caps.check_grid(n_stages, tp_size)
    if bad:
        raise SystemExit(f"[placement] {bad}")

    need = n_stages * tp_size
    if device_ids is None:
        # Warning: this is a shared machine — do not grab all cards by default.
        #   Take only as many as needed from the front; anything else must be
        #   given explicitly with --device-ids.
        device_ids = list(range(need))
    if len(device_ids) != need:
        raise SystemExit(f"[placement] --device-ids has {len(device_ids)} entries but "
                         f"PP={n_stages} × TP={tp_size} needs {need}")
    return [device_ids[k * tp_size:(k + 1) * tp_size] for k in range(n_stages)]


def fmt(devs: list[int], prefix: str = "rbln") -> str:
    if len(devs) == 1:
        return f"{prefix}{devs[0]}"
    return "{" + ",".join(f"{prefix}{d}" for d in devs) + "}"


def describe(stage_names, stage_devs, tp_size, prefix="rbln") -> str:
    body = ", ".join(f"{n}→{fmt(d, prefix)}" for n, d in zip(stage_names, stage_devs))
    return f"PP={len(stage_names)} × TP={tp_size} : {body}"


def check_shared_machine(verbose: bool = True) -> list[str]:
    """This is a shared machine — check for other users.

    Warning: cards may be in use by someone else even when the occupancy tool
      shows nothing.  Check `who` and process owners together — looking at only
      one is easy to misjudge.  If someone else holds a card, compilation
      succeeds and the run then fails with EBUSY during weight transfer, which
      is easily mistaken for a compiler problem.
    """
    others: list[str] = []
    try:
        me = subprocess.run(["id", "-un"], capture_output=True, text=True,
                            timeout=5).stdout.strip()
        who = subprocess.run(["who"], capture_output=True, text=True,
                             timeout=5).stdout
        users = {ln.split()[0] for ln in who.splitlines() if ln.split()}
        others = sorted(users - {me})
    except Exception:
        return []
    if verbose and others:
        print(f"[placement] Warning: shared machine — other users logged in: "
              f"{', '.join(others)}.  Leave cards free for long jobs "
              f"(--device-ids).", flush=True)
    return others
