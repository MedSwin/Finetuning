#!/usr/bin/env python3
"""
Benchmark local causal-LM models on OpenAI HealthBench using the SAME
text-generation flow and text-overlap metrics as bench_medquad.py.

Notes:
- HealthBench is rubric-oriented and does NOT always provide a single gold answer.
- To preserve MedQuAD-style metrics (Rouge-L / BERTScore / token-F1 /
  unigram precision / bigram precision), this script uses a single canonical
  text reference per example:
    1) ideal_completions_data.ideal_completion
    2) first non-empty ideal_completions_ref_completions entry
- Examples without a usable textual reference are skipped by default.
- This keeps metric logic aligned with bench_medquad.py, but it is NOT the
  official rubric-based HealthBench scoring method.

Example: run specific models
python scripts/bench_healthbench.py \
  --data-jsonl data/healthbench/healthbench_processed.jsonl \
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
  --batch-size 4 \
  --max-new-tokens 256
"""

import os, sys, json, time, argparse, random
from typing import List, Dict, Any, Optional

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_CACHE'] = '/fred/oz446/ModelBenchmarking/model'

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from bert_score import score as bertscore

torch.set_float32_matmul_precision('high')

INSTR = (
    "You are a careful clinical assistant. Answer the patient question using "
    "general, authoritative medical knowledge. Be concise (<=150 words), avoid speculation. "
    "If unsure, say: I don't know."
)


def norm_text(s):
    return " ".join((s or "").split()).strip()


def tok_f1(ref, cand):
    r = norm_text(ref).lower().split()
    c = norm_text(cand).lower().split()
    if not r and not c:
        return 1.0
    if not r or not c:
        return 0.0
    r_counts, c_counts = {}, {}
    for w in r:
        r_counts[w] = r_counts.get(w, 0) + 1
    for w in c:
        c_counts[w] = c_counts.get(w, 0) + 1
    overlap = sum(min(r_counts.get(w, 0), c_counts.get(w, 0)) for w in set(r_counts.keys()) | set(c_counts.keys()))
    prec = overlap / max(1, len(c))
    rec = overlap / max(1, len(r))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def ngram_precision(ref, cand, n=1):
    r = norm_text(ref).lower().split()
    c = norm_text(cand).lower().split()
    if len(c) < n:
        return 0.0

    def ngrams(x, n):
        return [" ".join(x[i:i + n]) for i in range(len(x) - n + 1)]

    rset = set(ngrams(r, n))
    cgrams = ngrams(c, n)
    if not cgrams:
        return 0.0
    hit = sum(1 for g in cgrams if g in rset)
    return hit / len(cgrams)


@torch.inference_mode()
def load_model(path, bf16_ok=True):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
            tok.pad_token = tok.eos_token or tok.convert_ids_to_tokens(tok.pad_token_id)
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
            tok.pad_token_id = tok.convert_tokens_to_ids("<|pad|>")
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
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tok))
    model.eval()
    return tok, model


def _content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_content_to_text(x) for x in value]
        return "\n".join([p for p in parts if norm_text(p)])
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if "content" in value:
            return _content_to_text(value["content"])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def extract_reference(obj: Dict[str, Any]) -> Optional[str]:
    """Pick one canonical reference so metrics match MedQuAD's single-ref flow."""
    ideal = obj.get("ideal_completions_data") or {}

    # 1) Primary ideal completion
    primary = norm_text(_content_to_text(ideal.get("ideal_completion")))
    if primary:
        return primary

    # 2) First usable reference completion
    for x in ideal.get("ideal_completions_ref_completions") or []:
        x = norm_text(_content_to_text(x))
        if x:
            return x

    # 3) Flat fallbacks for mixed processed/unprocessed exports
    for key in [
        "processed_ideal_completion_en_plaintext",
        "ideal_completion",
        "answer",
        "reference",
        "gold",
        "gold_answer",
    ]:
        val = norm_text(_content_to_text(obj.get(key)))
        if val:
            return val

    return None


def extract_messages(obj: Dict[str, Any]) -> List[Dict[str, str]]:
    candidates = [
        obj.get("prompt"),
        obj.get("messages"),
        obj.get("processed_prompt_en_plaintext"),
        obj.get("prompt_text"),
    ]

    for raw in candidates:
        cleaned: List[Dict[str, str]] = []

        if raw is None:
            continue

        if isinstance(raw, str):
            text = norm_text(raw)
            if text:
                return [{"role": "user", "content": text}]
            continue

        if isinstance(raw, dict):
            role = str(raw.get("role", "user")).strip().lower() or "user"
            content = norm_text(_content_to_text(raw.get("content", raw)))
            if content:
                return [{"role": role, "content": content}]
            continue

        if isinstance(raw, list):
            for m in raw:
                if isinstance(m, dict):
                    role = str(m.get("role", "user")).strip().lower() or "user"
                    content = norm_text(_content_to_text(m.get("content", "")))
                    if content:
                        cleaned.append({"role": role, "content": content})
                else:
                    content = norm_text(_content_to_text(m))
                    if content:
                        cleaned.append({"role": "user", "content": content})

            if cleaned:
                return cleaned

    return []


def convo_to_plaintext(messages: List[Dict[str, str]]) -> str:
    parts = []
    for m in messages:
        role = (m.get("role") or "user").strip().lower()
        if role == "assistant":
            label = "Assistant"
        elif role == "system":
            label = "System"
        else:
            label = "User"
        parts.append(f"{label}: {norm_text(m.get('content', ''))}")
    return "\n\n".join(parts).strip()


def format_prompt(messages: List[Dict[str, str]], tok: AutoTokenizer, system_msg: str = INSTR) -> str:
    cleaned = []
    if system_msg:
        cleaned.append({"role": "system", "content": system_msg})
    cleaned.extend(messages)

    if hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(cleaned, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    return f"{system_msg}\n\n{convo_to_plaintext(messages)}\n\nAssistant:"


@torch.inference_mode()
def generate_batch(tok, model, prompts, max_new_tokens=256):
    ctx = int(getattr(model.config, "max_position_embeddings", getattr(model.config, "n_positions", 4096)))
    max_inp = max(128, ctx - int(max_new_tokens) - 8)

    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_inp).to(model.device)

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
    empty_cnt = sum(1 for p in preds if len(p.strip()) == 0)
    print(f"[sanity] avg_pred_len={avg_len:.1f}w  empty={empty_cnt}/{len(preds)}")
    return preds


def load_healthbench_rows(path: str, max_samples: int, seed: int):
    rows = []
    skipped_no_ref = 0
    skipped_no_prompt = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            ref = extract_reference(obj)
            messages = extract_messages(obj)

            if not ref:
                skipped_no_ref += 1
                continue

            if not messages:
                skipped_no_prompt += 1
                continue

            rows.append({
                "id": obj.get("prompt_id") or obj.get("id") or f"row-{line_no}",
                "messages": messages,
                "prompt_text": convo_to_plaintext(messages),
                "ref": ref,
                "rubrics_json": json.dumps(obj.get("rubrics", []), ensure_ascii=False),
                "example_tags_json": json.dumps(obj.get("example_tags", []), ensure_ascii=False),
            })

    random.seed(seed)
    random.shuffle(rows)
    if max_samples and max_samples > 0 and max_samples < len(rows):
        rows = rows[:max_samples]

    print(
        f"[loader] usable={len(rows)} "
        f"skipped_no_ref={skipped_no_ref} "
        f"skipped_no_prompt={skipped_no_prompt}"
    )

    return rows, skipped_no_ref, skipped_no_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-jsonl", required=True, help="data/healthbench/healthbench_processed.jsonl")
    ap.add_argument("--model-dirs", nargs="+", required=True, help="e.g., model/medalpaca-7b model/medgemma-27b-text-it")
    ap.add_argument("--outdir", default="data/healthbench/runs")
    ap.add_argument("--max-samples", type=int, default=5000, help="cap for speed; set 0 for full usable split")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows, skipped_no_ref, skipped_no_prompt = load_healthbench_rows(
        args.data_jsonl, args.max_samples, args.seed
    )
    if not rows:
        raise RuntimeError(
            "No usable HealthBench rows found. Check that the JSONL includes prompt/messages "
            "and at least one textual reference (ideal_completion or fallback answer field)."
        )

    print(
        f"[data] usable={len(rows)}  skipped_no_ref={skipped_no_ref}  skipped_no_prompt={skipped_no_prompt}"
    )

    rscorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    summary_rows = []
    for mpath in args.model_dirs:
        mname = os.path.basename(mpath.rstrip("/"))
        print(f"\n=== Evaluating {mname} on {len(rows)} HealthBench items ===")

        tok, model = load_model(mpath)
        preds, refs, qids, prompt_texts, rubrics_all, tags_all = [], [], [], [], [], []

        for i in tqdm(range(0, len(rows), args.batch_size)):
            batch = rows[i:i + args.batch_size]
            prompts = [format_prompt(x["messages"], tok, INSTR) for x in batch]
            outs = generate_batch(tok, model, prompts, max_new_tokens=args.max_new_tokens)
            for b, gen in zip(batch, outs):
                preds.append(norm_text(gen))
                refs.append(norm_text(b["ref"]))
                qids.append(b["id"])
                prompt_texts.append(b["prompt_text"])
                rubrics_all.append(b["rubrics_json"])
                tags_all.append(b["example_tags_json"])

        rougeL_f, tokF1, uniP, biP = [], [], [], []
        for ref, hyp in zip(refs, preds):
            r = rscorer.score(ref, hyp)["rougeL"]
            rougeL_f.append(r.fmeasure)
            tokF1.append(tok_f1(ref, hyp))
            uniP.append(ngram_precision(ref, hyp, 1))
            biP.append(ngram_precision(ref, hyp, 2))

        P, R, F = bertscore(preds, refs, lang="en", rescale_with_baseline=False, model_type="roberta-large")
        bsf = F.tolist()

        df = pd.DataFrame({
            "id": qids,
            "prompt": prompt_texts,
            "ref": refs,
            "pred": preds,
            "rubrics": rubrics_all,
            "example_tags": tags_all,
            "rougeL_f": rougeL_f,
            "bert_f": bsf,
            "tok_f1": tokF1,
            "uni_prec": uniP,
            "bi_prec": biP,
        })
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_csv = os.path.join(args.outdir, f"{mname}_healthbench_{stamp}.csv")
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[saved] {out_csv}")

        summ = {
            "model": mname,
            "n": len(df),
            "skipped_no_ref_total": skipped_no_ref,
            "skipped_no_prompt_total": skipped_no_prompt,
            "rougeL_f_mean": float(df["rougeL_f"].mean()),
            "bert_f_mean": float(df["bert_f"].mean()),
            "tok_f1_mean": float(df["tok_f1"].mean()),
            "uni_prec_mean (halluc-proxy)": float(df["uni_prec"].mean()),
            "bi_prec_mean (halluc-proxy)": float(df["bi_prec"].mean()),
            "detail_csv": out_csv,
        }
        summary_rows.append(summ)

        del model
        del tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows)
    out_sum = os.path.join(args.outdir, f"SUMMARY_HEALTHBENCH_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    summary_df.to_csv(out_sum, index=False)
    print("\n=== SUMMARY ===")
    print(summary_df)
    print(f"[saved] {out_sum}")


if __name__ == "__main__":
    main()
