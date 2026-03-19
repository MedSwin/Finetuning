#!/usr/bin/env python3
'''
Benchmark MedMCQA with all default MedSwin benchmark models:

FULL default model cmd:
python scripts/bench_medmcqa.py \
  --data-jsonl data/medmcqa/medmcqa.jsonl \
  --use-default-models \
  --model-root model \
  --outdir output \
  --max-samples 0 \
  --batch-size 8 \
  --max-new-tokens 8

SINGLE model listed:
python scripts/bench_medmcqa.py \
  --data-jsonl data/medmcqa/medmcqa.jsonl \
  --model-dirs \
    model/BioMistral-7B \
    model/meditron-7b \
    model/medalpaca-7b \
    model/medalpaca-kd \
    model/medalpaca-sft \
    model/medalpaca-merged-task_arithmetic-sft-0.7 \
    model/medalpaca-merged-nuslerp-kd-0.7 \
    model/medalpaca-merged-dare_ties-kd-0.7-0.6 \
    model/medalpaca-merged-dare_linear-kd-0.7-0.55 \
    model/medalpaca-merged-ties-kd-0.7-0.6 \
    model/medalpaca-merged-dare_ties-kd-0.75-0.7 \
    model/medgemma-1.5 \
    model/medgemma-27b-text-it \
    model/medalpaca-kd-sft-pubmedqa-l-full \
    model/medalpaca-kd-sft-pubmedqa-map-full \
  --outdir output \
  --max-samples 0 \
  --batch-size 8 \
  --max-new-tokens 8
'''

import os
import json
import time
import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_CACHE"] = "/fred/oz446/ModelBenchmarking/model"

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_float32_matmul_precision("high")

SYS_MSG = (
    "You are a careful medical exam assistant. "
    "You must answer using only option letter(s). "
    "Never explain your reasoning. Never write full words unless they are already encoded as option letters."
)

DEFAULT_MODEL_NAMES = [
    "BioMistral-7B",
    "meditron-7b",
    "medalpaca-7b",
    "medalpaca-kd",
    "medalpaca-sft",
    "medalpaca-merged-task_arithmetic-sft-0.7",
    "medalpaca-merged-nuslerp-kd-0.7",
    "medalpaca-merged-dare_ties-kd-0.7-0.6",
    "medalpaca-merged-dare_linear-kd-0.7-0.55",
    "medalpaca-merged-ties-kd-0.7-0.6",
    "medalpaca-merged-dare_ties-kd-0.75-0.7",
    "medgemma-1.5",
    "medgemma-27b-text-it",
    "medalpaca-kd-sft-pubmedqa-l-full",
    "medalpaca-kd-sft-pubmedqa-map-full",
]

LETTER_BY_INDEX = {1: "a", 2: "b", 3: "c", 4: "d"}
VALID_LETTERS = ["a", "b", "c", "d"]
VALID_LETTER_SET = set(VALID_LETTERS)


def norm_text(s: Any) -> str:
    return " ".join(str(s or "").strip().split())


def normalize_option_text(s: Any) -> str:
    s = norm_text(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def normalize_choice_type(s: Any) -> str:
    s = norm_text(s).lower()
    return "multi" if s == "multi" else "single"


def uniq_sorted_letters(xs: List[str]) -> List[str]:
    return sorted({x.lower() for x in xs if x and x.lower() in VALID_LETTER_SET})


def cop_piece_to_letters(piece: Any, option_map: Dict[str, str]) -> List[str]:
    if piece is None:
        return []
    if isinstance(piece, int):
        return [LETTER_BY_INDEX[piece]] if piece in LETTER_BY_INDEX else []
    if isinstance(piece, float) and piece.is_integer():
        return [LETTER_BY_INDEX[int(piece)]] if int(piece) in LETTER_BY_INDEX else []
    if isinstance(piece, list):
        out: List[str] = []
        for x in piece:
            out.extend(cop_piece_to_letters(x, option_map))
        return uniq_sorted_letters(out)

    s = norm_text(piece)
    if not s:
        return []

    try:
        parsed = json.loads(s)
        if isinstance(parsed, (list, int, float, str)):
            return cop_piece_to_letters(parsed, option_map)
    except Exception:
        pass

    out: List[str] = []
    slow = s.lower()

    # Digits like "1", "1,3", "2 4"
    for d in re.findall(r"\b([1-4])\b", slow):
        out.append(LETTER_BY_INDEX[int(d)])

    # Letters like "a", "a,c", "b / d"
    out.extend(re.findall(r"\b([abcd])\b", slow))

    # Exact option text
    norm_s = normalize_option_text(s)
    for letter, text in option_map.items():
        if text and norm_s == normalize_option_text(text):
            out.append(letter)

    return uniq_sorted_letters(out)


def parse_gold_answers(obj: Dict[str, Any], option_map: Dict[str, str]) -> List[str]:
    gold = cop_piece_to_letters(obj.get("cop"), option_map)
    if gold:
        return gold
    for key in ["answer", "answers", "correct_option", "correct_options"]:
        if key in obj:
            gold = cop_piece_to_letters(obj.get(key), option_map)
            if gold:
                return gold
    return []


def build_user_prompt(row: Dict[str, Any]) -> str:
    q = norm_text(row["question"])
    a = norm_text(row["opa"])
    b = norm_text(row["opb"])
    c = norm_text(row["opc"])
    d = norm_text(row["opd"])

    if row["choice_type"] == "single":
        task = (
            "Task: choose the single best answer.\n"
            "Output format: exactly one lowercase letter: a or b or c or d.\n"
            "Return only the letter."
        )
    else:
        task = (
            "Task: choose all correct answers.\n"
            "Output format: lowercase letters only, sorted alphabetically, separated by commas.\n"
            "Valid examples: a,c   b,d   a,b,c\n"
            "Return only the letters."
        )

    return (
        f"Question: {q}\n\n"
        f"Options:\n"
        f"a. {a}\n"
        f"b. {b}\n"
        f"c. {c}\n"
        f"d. {d}\n\n"
        f"{task}\n\n"
        f"Answer:"
    )


def format_prompt(row: Dict[str, Any], tok: AutoTokenizer) -> str:
    user_prompt = build_user_prompt(row)
    if hasattr(tok, "apply_chat_template"):
        try:
            msgs = [
                {"role": "system", "content": SYS_MSG},
                {"role": "user", "content": user_prompt},
            ]
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"{SYS_MSG}\n\n{user_prompt}"


@torch.inference_mode()
def load_model(path: str, bf16_ok: bool = True):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)

    added_pad_token = False
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
            tok.pad_token = tok.eos_token or tok.convert_ids_to_tokens(tok.pad_token_id)
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
            tok.pad_token_id = tok.convert_tokens_to_ids("<|pad|>")
            added_pad_token = True

    tok.padding_side = "left"
    tok.truncation_side = "left"

    if bf16_ok and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if added_pad_token and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tok))

    model.eval()
    return tok, model


@torch.inference_mode()
def generate_batch(tok, model, prompts: List[str], max_new_tokens: int = 8) -> List[str]:
    ctx = int(getattr(model.config, "max_position_embeddings", getattr(model.config, "n_positions", 4096)))
    max_inp = max(128, ctx - int(max_new_tokens) - 8)

    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_inp,
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
    )

    gc = getattr(model, "generation_config", None)
    eos_id = getattr(gc, "eos_token_id", None)
    pad_id = getattr(gc, "pad_token_id", None)
    if eos_id is None and tok.eos_token_id is not None:
        eos_id = tok.eos_token_id
    if pad_id is None and tok.pad_token_id is not None:
        pad_id = tok.pad_token_id
    if eos_id is not None:
        gen_kwargs["eos_token_id"] = eos_id
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id

    out = model.generate(**enc, **gen_kwargs)

    attn = enc["attention_mask"]
    seqs = out.sequences
    preds = []
    for i in range(len(prompts)):
        in_len = int(attn[i].sum().item())
        gen_only_ids = seqs[i][in_len:]
        text = tok.decode(gen_only_ids, skip_special_tokens=True).strip()
        preds.append(text)

    avg_len = sum(len(p.split()) for p in preds) / max(1, len(preds))
    empty_cnt = sum(1 for p in preds if not p.strip())
    print(f"[sanity] avg_pred_len={avg_len:.2f}w  empty={empty_cnt}/{len(preds)}")
    return preds


def _extract_answer_segment(text: str) -> str:
    t = norm_text(text)
    if not t:
        return ""

    lower = t.lower()
    m = re.search(r"(?:^|\b)(?:final answer|answer|ans)\s*[:\-]?\s*(.+)$", lower)
    if m:
        seg = m.group(1).strip()
        if seg:
            return seg

    first_line = t.splitlines()[0].strip() if "\n" in t else t.strip()
    if first_line:
        return first_line
    return t


def _extract_letters_strict(segment: str) -> Optional[List[str]]:
    s = norm_text(segment).lower()
    if not s:
        return None

    s = s.replace("/", ",").replace(";", ",").replace("|", ",")
    s = re.sub(r"\band\b", ",", s)
    s = re.sub(r"\s+", " ", s).strip()

    patterns = [
        r"^(?:option\s+)?([abcd](?:\s*,\s*[abcd])*)$",
        r"^([abcd](?:\s*[,&/]\s*[abcd])*)$",
        r"^\(?([abcd])\)?$",
    ]
    for pat in patterns:
        m = re.match(pat, s)
        if m:
            letters = re.findall(r"[abcd]", m.group(1))
            return uniq_sorted_letters(letters)

    m = re.match(r"^(?:option\s+)?([1-4](?:\s*,\s*[1-4])*)$", s)
    if m:
        digits = re.findall(r"[1-4]", m.group(1))
        return uniq_sorted_letters([LETTER_BY_INDEX[int(d)] for d in digits])

    m = re.match(r"^(?:option\s+)?([1-4])$", s)
    if m:
        return [LETTER_BY_INDEX[int(m.group(1))]]

    return None


def _map_text_to_options(segment: str, option_map: Dict[str, str]) -> Tuple[List[str], str]:
    tnorm = normalize_option_text(segment)
    if not tnorm:
        return [], "empty"

    exact = []
    contained = []
    for letter, opt_text in option_map.items():
        onorm = normalize_option_text(opt_text)
        if not onorm:
            continue
        if tnorm == onorm:
            exact.append(letter)
        elif onorm in tnorm:
            contained.append(letter)

    if exact:
        return uniq_sorted_letters(exact), "option_text_exact"
    if len(contained) == 1:
        return uniq_sorted_letters(contained), "option_text_contained"
    return [], "unparsed"


def parse_prediction(raw_text: str, option_map: Dict[str, str], choice_type: str) -> Tuple[List[str], bool, str]:
    segment = _extract_answer_segment(raw_text)
    strict_letters = _extract_letters_strict(segment)
    if strict_letters is not None:
        if choice_type == "single":
            valid = len(strict_letters) == 1
            return strict_letters[:1], valid, "strict_letters"
        valid = len(strict_letters) >= 1
        return uniq_sorted_letters(strict_letters), valid, "strict_letters"

    mapped, source = _map_text_to_options(segment, option_map)
    if mapped:
        if choice_type == "single":
            valid = len(mapped) == 1
            return mapped[:1], valid, source
        valid = len(mapped) >= 1
        return uniq_sorted_letters(mapped), valid, source

    return [], False, source


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def pct(numer: int, denom: int) -> float:
    return round((100.0 * numer / denom), 4) if denom else 0.0


def ensure_model_dirs(args) -> List[str]:
    model_dirs: List[str] = []
    if args.use_default_models:
        model_dirs.extend([str(Path(args.model_root) / name) for name in DEFAULT_MODEL_NAMES])
    if args.model_dirs:
        model_dirs.extend(args.model_dirs)

    seen = set()
    ordered = []
    for m in model_dirs:
        if m not in seen:
            seen.add(m)
            ordered.append(m)

    if not ordered:
        raise ValueError("No models selected. Use --use-default-models or provide --model-dirs.")

    missing = [m for m in ordered if not Path(m).exists()]
    if missing:
        raise FileNotFoundError("Missing model directories:\n  " + "\n  ".join(missing))

    return ordered


def build_run_dir(outdir: str, run_name: Optional[str]) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = run_name or f"medmcqa_{stamp}"
    run_dir = Path(outdir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-jsonl", required=True, help="data/medmcqa/medmcqa.jsonl")
    ap.add_argument("--model-dirs", nargs="+", default=None, help="Explicit model directories")
    ap.add_argument("--use-default-models", action="store_true", help="Benchmark all default model names under --model-root")
    ap.add_argument("--model-root", default="model", help="Root folder containing benchmark model directories")
    ap.add_argument("--outdir", default="output", help="Root output folder")
    ap.add_argument("--run-name", default=None, help="Optional run folder name inside outdir")
    ap.add_argument("--max-samples", type=int, default=5000, help="Cap for speed; set 0 or negative for full dataset")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    args = ap.parse_args()

    random.seed(args.seed)
    model_dirs = ensure_model_dirs(args)
    run_dir = build_run_dir(args.outdir, args.run_name)

    rows = []
    skipped = 0
    with open(args.data_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            option_map = {
                "a": norm_text(obj.get("opa")),
                "b": norm_text(obj.get("opb")),
                "c": norm_text(obj.get("opc")),
                "d": norm_text(obj.get("opd")),
            }
            choice_type = normalize_choice_type(obj.get("choice_type"))
            gold_letters = parse_gold_answers(obj, option_map)

            if not obj.get("question") or not all(option_map.values()) or not gold_letters:
                skipped += 1
                continue

            rows.append({
                "id": obj.get("id"),
                "question": obj.get("question"),
                "opa": option_map["a"],
                "opb": option_map["b"],
                "opc": option_map["c"],
                "opd": option_map["d"],
                "subject_name": norm_text(obj.get("subject_name")),
                "topic_name": norm_text(obj.get("topic_name")),
                "choice_type": choice_type,
                "gold_letters": gold_letters,
                "gold_texts": [option_map[x] for x in gold_letters],
            })

    random.shuffle(rows)
    if args.max_samples and args.max_samples > 0 and args.max_samples < len(rows):
        rows = rows[:args.max_samples]

    dataset_stats = {
        "data_jsonl": args.data_jsonl,
        "loaded_rows": len(rows),
        "skipped_rows": skipped,
        "single_rows": sum(1 for x in rows if x["choice_type"] == "single"),
        "multi_rows": sum(1 for x in rows if x["choice_type"] == "multi"),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "model_dirs": model_dirs,
    }
    with open(run_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, indent=2, ensure_ascii=False)

    print(f"[data] loaded={len(rows)} skipped={skipped} run_dir={run_dir}")

    summary_rows = []
    for mpath in model_dirs:
        mname = Path(mpath).name
        model_dir = run_dir / mname
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Evaluating {mname} on {len(rows)} items ===")

        model_start = time.time()
        tok, model = load_model(mpath)
        results = []

        for i in tqdm(range(0, len(rows), args.batch_size), desc=mname):
            batch = rows[i:i + args.batch_size]
            prompts = [format_prompt(x, tok) for x in batch]
            outs = generate_batch(tok, model, prompts, max_new_tokens=args.max_new_tokens)

            for row, raw_pred in zip(batch, outs):
                option_map = {
                    "a": row["opa"],
                    "b": row["opb"],
                    "c": row["opc"],
                    "d": row["opd"],
                }
                pred_letters, is_valid, parse_source = parse_prediction(raw_pred, option_map, row["choice_type"])
                pred_texts = [option_map[x] for x in pred_letters if x in option_map]
                exact_match = set(pred_letters) == set(row["gold_letters"])

                if not is_valid:
                    outcome = "invalid"
                elif exact_match:
                    outcome = "correct"
                else:
                    outcome = "incorrect"

                results.append({
                    "id": row["id"],
                    "subject_name": row["subject_name"],
                    "topic_name": row["topic_name"],
                    "choice_type": row["choice_type"],
                    "question": row["question"],
                    "opa": row["opa"],
                    "opb": row["opb"],
                    "opc": row["opc"],
                    "opd": row["opd"],
                    "gold_letters": ",".join(row["gold_letters"]),
                    "gold_texts": " | ".join(row["gold_texts"]),
                    "pred_raw": norm_text(raw_pred),
                    "pred_letters": ",".join(pred_letters),
                    "pred_texts": " | ".join(pred_texts),
                    "parse_source": parse_source,
                    "is_valid": int(is_valid),
                    "correct": int(exact_match),
                    "outcome": outcome,
                })

        df = pd.DataFrame(results)
        audit_csv = model_dir / "audit.csv"
        df.to_csv(audit_csv, index=False, encoding="utf-8")

        total_n = int(len(df))
        correct_n = int((df["outcome"] == "correct").sum())
        incorrect_n = int((df["outcome"] == "incorrect").sum())
        invalid_n = int((df["outcome"] == "invalid").sum())
        df_single = df[df["choice_type"] == "single"]
        df_multi = df[df["choice_type"] == "multi"]
        elapsed_sec = round(time.time() - model_start, 4)

        metrics = {
            "model": mname,
            "model_dir": mpath,
            "n": total_n,
            "correct_count": correct_n,
            "incorrect_count": incorrect_n,
            "invalid_count": invalid_n,
            "accuracy_pct": pct(correct_n, total_n),
            "valid_rate_pct": pct(total_n - invalid_n, total_n),
            "single_n": int(len(df_single)),
            "single_correct_count": int((df_single["outcome"] == "correct").sum()),
            "single_incorrect_count": int((df_single["outcome"] == "incorrect").sum()),
            "single_invalid_count": int((df_single["outcome"] == "invalid").sum()),
            "single_accuracy_pct": pct(int((df_single["outcome"] == "correct").sum()), int(len(df_single))),
            "multi_n": int(len(df_multi)),
            "multi_correct_count": int((df_multi["outcome"] == "correct").sum()),
            "multi_incorrect_count": int((df_multi["outcome"] == "incorrect").sum()),
            "multi_invalid_count": int((df_multi["outcome"] == "invalid").sum()),
            "multi_accuracy_pct": pct(int((df_multi["outcome"] == "correct").sum()), int(len(df_multi))),
            "elapsed_sec": elapsed_sec,
            "audit_csv": str(audit_csv),
        }

        with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        summary_rows.append(metrics)
        print(f"[saved] {audit_csv}")
        print(
            f"[metrics] model={mname} accuracy={metrics['accuracy_pct']:.2f}% "
            f"correct={correct_n} incorrect={incorrect_n} invalid={invalid_n}"
        )

        del model
        del tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = run_dir / "summary.csv"
    summary_json = run_dir / "summary.json"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    run_config = {
        "command_args": vars(args),
        "resolved_model_dirs": model_dirs,
        "run_dir": str(run_dir),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    if not summary_df.empty:
        print(summary_df[[
            "model", "n", "correct_count", "incorrect_count", "invalid_count", "accuracy_pct"
        ]].to_string(index=False))
    print(f"[saved] {summary_csv}")
    print(f"[saved] {summary_json}")


if __name__ == "__main__":
    main()
