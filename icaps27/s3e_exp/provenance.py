# SPDX-License-Identifier: MIT
"""What every run records so it can be reported and reproduced.

Follows the ICAPS 2027 reproducibility requirements: library versions,
hardware and OS, model identifiers and resolved revisions, decoding
parameters, prompts, and the dates on which models were queried. The host
name and user name are deliberately not recorded (anonymous artifact).
"""

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPORTED_PACKAGES = (
    "s3e", "torch", "transformers", "accelerate", "vllm", "openai", "unified-planning",
    "up-fast-downward", "scikit-learn", "numpy", "Pillow", "t2v-metrics",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_versions() -> dict:
    versions = {}
    for name in REPORTED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def s3e_source() -> dict:
    """Where s3e was imported from and its version (installed vs overlay)."""
    import s3e

    return {"version": getattr(s3e, "__version__", None),
            "path": str(Path(s3e.__file__).resolve().parent)}


def experiments_commit() -> "dict | None":
    """The commit of this experiments directory, if it is a git checkout."""
    here = Path(__file__).resolve().parent

    def git(*args):
        return subprocess.run(["git", *args], cwd=here, capture_output=True, text=True,
                              check=True).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--", str(here.parent)))}
    except (OSError, subprocess.CalledProcessError):
        return None


def hardware() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
    }
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return info


def model_revision(backend, spec: dict) -> "str | None":
    """The Hugging Face commit the model was loaded from, when knowable."""
    inner = getattr(backend, "inner", backend)
    config = getattr(getattr(inner, "model", None), "config", None)
    revision = getattr(config, "_commit_hash", None)
    if revision:
        return revision
    if spec.get("backend") in ("openai", "fake"):
        return None
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(spec["id"], revision=spec.get("revision"),
                                     local_files_only=True)
    except Exception:  # not a cached Hub model: provenance stays unknown
        return spec.get("revision")
    return Path(snapshot).name


def peak_memory() -> dict:
    """Peak CUDA memory on device 0 since the last reset, else peak RSS."""
    try:
        import torch

        if torch.cuda.is_available():
            return {"kind": "cuda_max_memory_allocated", "bytes": torch.cuda.max_memory_allocated()}
    except ImportError:
        pass
    try:
        import resource
    except ImportError:
        return {"kind": "unavailable", "bytes": None}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"kind": "process_peak_rss",
            "bytes": peak if sys.platform == "darwin" else peak * 1024}


def reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def base(argv: "list[str] | None" = None) -> dict:
    return {
        "started_utc": now(),
        "argv": list(sys.argv if argv is None else argv),
        "packages": package_versions(),
        "s3e": s3e_source(),
        "experiments_commit": experiments_commit(),
        "hardware": hardware(),
    }
