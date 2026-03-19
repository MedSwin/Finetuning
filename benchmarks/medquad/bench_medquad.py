#!/usr/bin/env python3
import os, sys, json, math, time, argparse, random
from typing import List

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
    if not r and not c: return 1.0
    if not r or not c:  return 0.0
    r_counts, c_counts = {}, {}
    for w in r: r_counts[w] = r_counts.get(w, 0) + 1
    for w in c: c_counts[w] = c_counts.get(w, 0) + 1
    overlap = sum(min(r_counts.get(w,0), c_counts.get(w,0)) for w in set(r_counts.keys()) | set(c_counts.keys()))
    prec = overlap / max(1, len(c))
    rec  = overlap / max(1, len(r))
    if prec+rec == 0: return 0.0
    return 2*prec*rec/(prec+rec)

def ngram_precision(ref, cand, n=1):
    r = norm_text(ref).lower().split()
    c = norm_text(cand).lower().split()
    if len(c) < n: return 0.0
    def ngrams(x, n): return [" ".join(x[i:i+n]) for i in range(len(x)-n+1)]
    rset = set(ngrams(r, n))
    cgrams = ngrams(c, n)
    if not cgrams: return 0.0
    hit = sum(1 for g in cgrams if g in rset)
    return hit / len(cgrams)

@torch.inference_mode()
def load_model(path, bf16_ok=True):
    tok = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=True
    )
    # If pad_token_id is missing, use eos (typical for LLaMA-family)
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
            tok.pad_token = tok.eos_token or tok.convert_ids_to_tokens(tok.pad_token_id)
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
            tok.pad_token_id = tok.convert_tokens_to_ids("<|pad|>")
    tok.padding_side = "left"
    tok.truncation_side = "left"

    # Choose a safe dtype for GPUs/CPU
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
    # Ensure lm_head knows about newly added pad token (rare, but safe)
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tok))
    model.eval()
    return tok, model

def format_prompt(question: str, tok: AutoTokenizer, system_msg: str = INSTR) -> str:
    q = norm_text(question)
    # If the tokenizer ships a chat template, use it.
    if hasattr(tok, "apply_chat_template"):
        try:
            msgs = [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": q},
            ]
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    # Fallback (no template available)
    return f"{system_msg}\n\nQuestion: {q}\n\nAnswer:"

@torch.inference_mode()
def generate_batch(tok, model, prompts, max_new_tokens=256):
    # keep input within context
    ctx = int(getattr(model.config, "max_position_embeddings",
                      getattr(model.config, "n_positions", 4096)))
    max_inp = max(128, ctx - int(max_new_tokens) - 8)
    
    # Tokenize with left-padding; keep attention_mask for true lengths
    enc = tok(
        prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=max_inp
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,  # deterministic for eval
        use_cache=True,
        return_dict_in_generate=True,
    )
    
    # Only pass IDs that actually exist in generation_config
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

    # Format generation on enc + kwargs
    out = model.generate(**enc, **gen_kwargs)
    
    # Robustly slice off the prompt by token lengths (not string prefixes)
    # With left padding, true input length = sum(attention_mask[i])
    attn = enc["attention_mask"]
    seqs = out.sequences
    preds = []
    for i in range(len(prompts)):
        in_len = int(attn[i].sum().item())
        # sequences[i] is [prompt_tokens ... generated_tokens]
        gen_only_ids = seqs[i][in_len:]
        text = tok.decode(gen_only_ids, skip_special_tokens=True).strip()
        preds.append(text)
    avg_len = sum(len(p.split()) for p in preds) / max(1, len(preds))
    empty_cnt = sum(1 for p in preds if len(p.strip()) == 0)
    print(f"[sanity] avg_pred_len={avg_len:.1f}w  empty={empty_cnt}/{len(preds)}")
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-jsonl", required=True, help="data/medquad/processed/medquad_clean.jsonl")
    ap.add_argument("--model-dirs", nargs="+", required=True, help="e.g., model/medalpaca-7b model/medgemma-27b-text-it")
    ap.add_argument("--outdir", default="data/medquad/runs")
    ap.add_argument("--max-samples", type=int, default=5000, help="cap for speed; set higher to run full")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # Load dataset
    rows = []
    with open(args.data_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append({"id": obj["id"], "q": obj["question"], "a": obj["answer"]})
    random.shuffle(rows)
    if args.max_samples and args.max_samples < len(rows):
        rows = rows[:args.max_samples]

    # Prepare metric tools
    rscorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # Evaluate each model
    summary_rows = []
    for mpath in args.model_dirs:
        mname = os.path.basename(mpath.rstrip("/"))
        print(f"\n=== Evaluating {mname} on {len(rows)} items ===")

        tok, model = load_model(mpath)
        preds, refs, qids = [], [], []

        # Batched generation
        for i in tqdm(range(0, len(rows), args.batch_size)):
            batch = rows[i:i+args.batch_size]
            # Prompt builder with token-aware 
            prompts = [format_prompt(x["q"], tok, INSTR) for x in batch]
            outs = generate_batch(tok, model, prompts, max_new_tokens=args.max_new_tokens)
            for b, gen in zip(batch, outs):
                preds.append(norm_text(gen))
                refs.append(norm_text(b["a"]))
                qids.append(b["id"])

        # Metrics: Rouge-L, token F1, n-gram precision
        rougeL_f, tokF1, uniP, biP = [], [], [], []
        for ref, hyp in zip(refs, preds):
            r = rscorer.score(ref, hyp)["rougeL"]
            rougeL_f.append(r.fmeasure)
            tokF1.append(tok_f1(ref, hyp))
            uniP.append(ngram_precision(ref, hyp, 1))
            biP.append(ngram_precision(ref, hyp, 2))

        # BERTScore (batched) + Safe default rescale=true; robust rescale=True
        P, R, F = bertscore(preds, refs, lang="en", rescale_with_baseline=False, model_type="roberta-large")
        bsf = F.tolist()

        df = pd.DataFrame({
            "id": qids, "ref": refs, "pred": preds,
            "rougeL_f": rougeL_f, "bert_f": bsf,
            "tok_f1": tokF1, "uni_prec": uniP, "bi_prec": biP
        })
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_csv = os.path.join(args.outdir, f"{mname}_medquad_{stamp}.csv")
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[saved] {out_csv}")

        # Summary
        summ = {
            "model": mname,
            "n": len(df),
            "rougeL_f_mean": float(df["rougeL_f"].mean()),
            "bert_f_mean": float(df["bert_f"].mean()),
            "tok_f1_mean": float(df["tok_f1"].mean()),
            "uni_prec_mean (halluc-proxy)": float(df["uni_prec"].mean()),
            "bi_prec_mean (halluc-proxy)": float(df["bi_prec"].mean()),
            "detail_csv": out_csv
        }
        summary_rows.append(summ)

        # free VRAM between models
        del model; del tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write overall summary
    summary_df = pd.DataFrame(summary_rows)
    out_sum = os.path.join(args.outdir, f"SUMMARY_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    summary_df.to_csv(out_sum, index=False)
    print("\n=== SUMMARY ===")
    print(summary_df)
    print(f"[saved] {out_sum}")

if __name__ == "__main__":
    main()


