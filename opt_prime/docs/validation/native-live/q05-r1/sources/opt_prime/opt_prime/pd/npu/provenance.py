"""Small reproducibility records without weights, binaries or environment secrets."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time


def record_run(directory, script):
    package = Path(__file__).resolve().parents[1]
    repo = Path(__file__).resolve().parents[4]
    files = sorted(package.rglob("*.py")) + [Path(script).resolve()]
    sources = {}
    for path in files:
        relative = str(path.relative_to(repo))
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        target = Path(directory) / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    versions = {}
    for name in (
        "torch",
        "furiosa-torch",
        "furiosa-llm",
        "rebel-compiler",
        "torch-rbln",
        "optimum-rbln",
        "transformers",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    record = dict(
        argv=sys.argv,
        python=sys.executable,
        cwd=str(Path.cwd()),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        versions=versions,
        sources=sources,
        commit=subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        status=subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True
        ),
    )
    (Path(directory) / "provenance.json").write_text(json.dumps(record, indent=2))
