# SPDX-License-Identifier: MIT
"""The answer-probability code research projects typically write by hand.

A deliberately plain reimplementation of what VLM-as-grounder code usually
does to read P(yes) (as in VQAScore-style scorers): one forward pass per
query, the softmax at the last position, and the probability of the single
token id of "Yes" (and of "No"), with no case or leading-space variants, no
batching, no check that the token is the model's answer at all, and no
handling of thinking blocks. It imports no S3E code; E6 compares it with
S3E's logprobs mode on the same prompts, model, and images.

    python s3e_exp/workers/adhoc_worker.py queries.jsonl out.jsonl \\
        --model Qwen/Qwen3-VL-8B-Instruct --revision <commit>
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("queries", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--yes", default="Yes")
    parser.add_argument("--no", default="No")
    args = parser.parse_args(argv)

    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, revision=args.revision, dtype=getattr(torch, args.dtype), device_map="auto")
    model.eval()
    tokenizer = processor.tokenizer
    yes_id = tokenizer.convert_tokens_to_ids(args.yes)
    no_id = tokenizer.convert_tokens_to_ids(args.no)

    rows = [json.loads(line) for line in args.queries.open(encoding="utf-8") if line.strip()]
    done = set()
    if args.out.exists():
        done = {json.loads(line)["key"] for line in args.out.open(encoding="utf-8") if line.strip()}
    with args.out.open("a", encoding="utf-8") as out:
        for row in rows:
            if row["key"] in done:
                continue
            start = time.perf_counter()
            messages = []
            if row.get("system_prompt"):
                messages.append({"role": "system",
                                 "content": [{"type": "text", "text": row["system_prompt"]}]})
            messages.append({"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": row["prompt"]}]})
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            with Image.open(row["image"]) as image:
                inputs = processor(text=[text], images=[image.convert("RGB")],
                                   return_tensors="pt").to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
            p_yes, p_no = float(probs[yes_id]), float(probs[no_id])
            top = int(torch.argmax(probs))
            total = p_yes + p_no
            out.write(json.dumps({
                "key": row["key"], "p_yes_token": p_yes, "p_no_token": p_no,
                "probability": p_yes / total if total > 0 else 0.5, "has_answer": total > 0,
                "argmax_is_answer": top in (yes_id, no_id), "seconds": time.perf_counter() - start,
            }) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
