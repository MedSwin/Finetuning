#!/usr/bin/env python3
# scripts/demo.py
#
# Run a fixed evaluation prompt across multiple local HF models (BF16, no quantization),
# and write outputs to a chat-like boxed log file.
#
# Usage:
#   python scripts/demo.py
#   python scripts/demo.py --log_file logs/medswin_demo.log --max_new_tokens 256 --temperature 0.2
#
# Requirements:
#   pip install -U torch transformers accelerate

import os
import gc
import time
import argparse
import textwrap
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_SPECS = [
    ("medalpaca-7b", "MedAlpaca Base"),
    ("medalpaca-sft", "MedSwin SFT"),
    ("medalpaca-kd", "MedSwin KD"),
    ("medalpaca-kd-sft-pubmedqa-map-full", "MedSwin KD-SFT"),
    ("medalpaca-merged-dare_ties-kd-0.7-0.6", "DaRE-TIES-KD-0.7"),
    ("medalpaca-merged-nuslerp-kd-0.7", "NuSLERP-KD-0.7"),
    ("medalpaca-merged-task_arithmetic-kd-w_0.7", "TA-KD-0.7"),
]


DEFAULT_PROMPT = """You are a clinical assistant.
Answer concisely and focus on key clinical reasoning.
Do NOT invent facts; if unsure, say “uncertain”. Use generic drug classes (not brand names).
Include: (1) top 3 differentials with 1 reason each, (2) red flags requiring urgent care,
(3) immediate workup/management in the first hour, (4) 2-sentence patient-friendly explanation.

Case: 58M, sudden shortness of breath + pleuritic chest pain. HR 112, SpO2 90% (room air).
Recent 12-hour flight 2 days ago; mild unilateral calf swelling. No fever or cough.
PMH: hypertension on amlodipine. No known allergies.
"""


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def wrap_lines(s: str, width: int):
    # Preserve paragraphs while wrapping
    out = []
    for para in s.splitlines():
        if not para.strip():
            out.append("")
        else:
            out.extend(textwrap.wrap(para, width=width, break_long_words=False, replace_whitespace=False))
    return out


def boxed_chat_block(role_left: str, role_right: str, content: str, width: int = 110) -> str:
    """
    Draw a chat-like box using '_' and '|'.
    Example:
     ____________________________________________
    | USER                                       |
    | ...                                        |
    |--------------------------------------------|
    | ASSISTANT (X Model)                        |
    | ...                                        |
    |____________________________________________|
    """
    # Box geometry
    inner_w = max(40, width)
    top = " " + "_" * (inner_w + 2)
    bot = "|" + "_" * (inner_w + 2) + "|"

    def line(text=""):
        t = (text or "")[:inner_w]
        return "| " + t.ljust(inner_w) + " |"

    sep = "|" + "-" * (inner_w + 2) + "|"

    lines = [top]
    # Header left
    hdr_left = f"{role_left}".strip()
    lines.append(line(hdr_left))
    lines.append(sep)

    # Content left
    for wline in wrap_lines(content.strip(), width=inner_w):
        lines.append(line(wline))

    lines.append(sep)
    # Header right
    hdr_right = f"{role_right}".strip()
    lines.append(line(hdr_right))
    lines.append(sep)

    # Placeholder for assistant content filled by caller (we return up to assistant header)
    return "\n".join(lines)


def boxed_assistant_only(role_right: str, content: str, width: int = 110) -> str:
    inner_w = max(40, width)
    sep = "|" + "-" * (inner_w + 2) + "|"
    bot = "|" + "_" * (inner_w + 2) + "|"

    def line(text=""):
        t = (text or "")[:inner_w]
        return "| " + t.ljust(inner_w) + " |"

    lines = []
    # Assistant content
    for wline in wrap_lines(content.strip(), width=inner_w):
        lines.append(line(wline))
    lines.append(bot)
    return "\n".join(lines)


@torch.inference_mode()
def generate_one(model_path: Path, prompt: str, args) -> str:
    # Enforce BF16 (no quantization)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script is intended for HPC GPUs.")

    # (Optional) sanity check for BF16 support
    # A100/H100/etc support BF16; some older GPUs may not.
    if args.require_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not report BF16 support, but BF16 is required by args.require_bf16.")

    tok = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )

    # Some LLaMA-like tokenizers don't have pad_token by default
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    # Prefer device_map="auto" for sharding if multiple GPUs exist
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
    )
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()

    # Tokenize with a strict max input budget (keep under 1024 total context)
    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    )

    # Move tensors to the correct device when not using device_map sharding
    # If device_map is "auto", the model is sharded and inputs can stay on CPU; transformers will handle placement.
    if args.device_map in ("cpu", None):
        device = torch.device("cuda:0")
        inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,                 # use low temp sampling (requested)
        temperature=args.temperature,   # low temperature
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        use_cache=True,
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    out = model.generate(**inputs, **gen_kwargs)

    # Slice the generated continuation only (remove the prompt tokens)
    prompt_len = inputs["input_ids"].shape[-1]
    gen_ids = out[0][prompt_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()

    # Cleanup
    del model
    del tok
    gc.collect()
    torch.cuda.empty_cache()

    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", type=str, default=None, help="Path to 'model' directory (default: <repo>/model)")
    ap.add_argument("--log_file", type=str, default=None, help="Output log file path (default: <repo>/logs/demo.log)")
    ap.add_argument("--prompt_file", type=str, default=None, help="Optional path to a txt file containing the prompt")
    ap.add_argument("--max_input_tokens", type=int, default=768, help="Max input tokens (keep total under 1024)")
    ap.add_argument("--max_new_tokens", type=int, default=256, help="Max generation tokens")
    ap.add_argument("--temperature", type=float, default=0.2, help="Low temperature to reduce hallucination")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device_map", type=str, default="auto", help='Usually "auto". Use "cpu" to disable sharding.')
    ap.add_argument("--trust_remote_code", action="store_true", help="Enable if your model repo requires it")
    ap.add_argument("--attn_implementation", type=str, default=None,
                    help='Optional: "flash_attention_2" if installed, else leave empty.')
    ap.add_argument("--require_bf16", action="store_true",
                    help="Fail if GPU does not support BF16 (recommended for strict compliance).")
    ap.add_argument("--box_width", type=int, default=110)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    models_dir = Path(args.models_dir) if args.models_dir else (repo_root / "model")
    logs_dir = repo_root / "logs"
    ensure_dir(logs_dir)

    log_file = Path(args.log_file) if args.log_file else (logs_dir / "demo.log")

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    else:
        prompt = DEFAULT_PROMPT

    header = []
    header.append("=" * 100)
    header.append(f"MedSwin Demo Run @ {now_str()}")
    header.append(f"models_dir: {models_dir}")
    header.append(f"gen: max_input_tokens={args.max_input_tokens}, max_new_tokens={args.max_new_tokens}, "
                  f"temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, "
                  f"repetition_penalty={args.repetition_penalty}, seed={args.seed}, device_map={args.device_map}, "
                  f"bf16_required={args.require_bf16}")
    header.append("=" * 100)
    header.append("")
    header_text = "\n".join(header)

    # Write header once (append mode so repeated runs accumulate)
    ensure_dir(log_file.parent)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(header_text + "\n")

    for folder, display_name in MODEL_SPECS:
        model_path = models_dir / folder
        if not model_path.exists():
            msg = f"[WARN] Missing model path: {model_path} (skipping)\n"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg)
            continue

        # Prepare box skeleton
        role_left = "USER"
        role_right = f"ASSISTANT ({display_name})"
        box_top = boxed_chat_block(role_left, role_right, prompt, width=args.box_width)

        start = time.time()
        try:
            answer = generate_one(model_path, prompt, args)
            elapsed = time.time() - start
            meta = f"[{display_name}] path={model_path} | time={elapsed:.2f}s | {now_str()}"
            block = box_top + "\n" + boxed_assistant_only(role_right, answer, width=args.box_width) + "\n" + meta + "\n\n"
        except Exception as e:
            elapsed = time.time() - start
            meta = f"[{display_name}] path={model_path} | time={elapsed:.2f}s | {now_str()}"
            err_text = f"ERROR: {type(e).__name__}: {e}"
            block = box_top + "\n" + boxed_assistant_only(role_right, err_text, width=args.box_width) + "\n" + meta + "\n\n"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(block)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n\n")

    print(f"Done. Log written to: {log_file}")


if __name__ == "__main__":
    main()

