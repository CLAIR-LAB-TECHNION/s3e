# SPDX-License-Identifier: MIT
"""Query a model on an instance set and store raw S3E predictions.

This is the only expensive stage shared by E1 (offline planning), E3
(scoring modes), and E4 (calibration and views): it writes
``<out>/<domain>/<model>/<scoring>-<mask>/predictions.jsonl`` (one record
per scene and view, with every prediction's raw answer masses) and
``run.json`` (provenance: versions, hardware, model revision, prompts,
answer tokens, decoding settings, query dates). Reruns resume.

Example (a GPU is assumed for anything but ``--model fake``)::

    python experiments/collect.py --instances instances/blocksworld \\
        --model qwen3-vl-8b --scoring logprobs --views 0 1 2 3 all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s3e_exp import cli, domains  # noqa: E402
from s3e_exp.estimation import EstimationConfig, collect  # noqa: E402
from s3e_exp.masking import MASKS  # noqa: E402
from s3e_exp.models import build_backend, model_spec  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    cli.scene_args(p)
    cli.model_args(p)
    p.add_argument("--scoring", choices=["logprobs", "text_match"], default="logprobs")
    p.add_argument("--mask", choices=MASKS, default="none")
    p.add_argument("--views", nargs="+", default=["0"],
                   help="view specs: an index, 'all' (every view in one prompt), or '0,2'")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=8,
                   help="reply cap for text scoring (ignored for logprobs)")
    p.add_argument("--null-tokens", nargs="*", default=None,
                   help="tokens of an explicit 'unknown' answer, e.g. unknown")
    p.add_argument("--system-prompt", default=None,
                   help="override S3E's system prompt ('' sends none, as t2v_metrics does)")
    p.add_argument("--tag", default=None, help="suffix for the run directory")
    p.add_argument("--out", type=Path, default=Path("results/raw/predictions"))
    return p


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    scenes = cli.select_scenes(args)
    domain_name = cli.domain_of(scenes)
    spec = model_spec(args.model)
    backend = build_backend(spec, args.backend, **args.vlm_kwargs)
    config = EstimationConfig(
        scoring=args.scoring, mask=args.mask, views=tuple(args.views),
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        null_tokens=args.null_tokens, system_prompt=args.system_prompt,
    )
    name = config.tag() + (f"-{args.tag}" if args.tag else "")
    if args.backend and args.backend != spec.get("backend", "hf"):
        name += f"-{args.backend}"
    out = cli.run_dir(args.out, domain_name, spec["name"], name)
    collect(scenes, domains.get(domain_name), spec, backend, config, out,
            extra_provenance={"instances": str(args.instances), "scenes": len(scenes),
                              "vlm_kwargs": args.vlm_kwargs, "backend": args.backend})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
