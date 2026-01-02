#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pipeline.py — Distill (text/soft) from a local teacher, then LoRA/QLoRA finetune.
Outputs audit-friendly artifacts: per-dataset distilled JSONLs, merged JSONL,
(optional) .gz soft-labels, a manifest, (optional) split JSONLs, logs, best checkpoint.

Subcommands:
  distill   : run teacher generation over 1..N datasets
  finetune  : train LoRA/QLoRA on distilled JSONL(s)
  all       : distill then finetune

Helper:
  python scripts/pipeline.py distill --help
  python scripts/pipeline.py finetune --help

Example:
python scripts/longdistill_finetune.py all \
  --command soft \
  --data-files "data/healthcaremagic.jsonl,data/pubmedqa_map.jsonl,data/pubmedqa_u.csv,data/pubmedqa_l.csv" \
  --teacher-dir model/medgemma-27b-text-it \
  --out-dir artifacts/all_2025-11-09 \
  --max-new-tokens 1024 \
  --temperature 0.0 \
  --batch-size 4 \
  --topk-logprobs 10 \
  --model-dir model/medalpaca-7b \
  --use-qlora --bf16 --gradient-checkpointing \
  --epochs 2 --batch-size-ft 1 --grad-accum 16 \
  --eval-steps 200 --save-steps 200 --logging-steps 50 \
  --save-splits
"""

import os, sys, re, json, gzip, glob, time, math, random, argparse, traceback, csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

import numpy as np
import torch
torch.set_float32_matmul_precision('high')

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
)
from transformers.trainer_callback import EarlyStoppingCallback

# peft/qlora
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel


# -------- logging --------
def setup_file_logger(path: Union[str, Path]):
    path = str(path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path, mode="w", encoding="utf-8")]
    )
    return logging.getLogger(__name__)


# -------- I/O helpers --------
def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p


def read_jsonl(path: Union[str, Path]):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)


def write_jsonl(rows: List[Dict[str, Any]], path: Union[str, Path]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_jsonl_line(path: Union[str, Path], row: Dict[str, Any]):
     """Append a single JSON row to JSONL (creates file if missing)."""
     with open(path, "a", encoding="utf-8") as f:
         f.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_jsonl_any(path: Union[str, Path], max_lines: Optional[int] = None) -> List[Dict[str, Any]]:
    p = str(path)
    opener = gzip.open if p.endswith(".gz") else open
    rows: List[Dict[str, Any]] = []
    n = 0
    try:
        with opener(p, "rt", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s: continue
                try:
                    rows.append(json.loads(s))
                except json.JSONDecodeError:
                    break  # tolerate partial final line
                n += 1
                if max_lines and n >= max_lines:
                    break
    except (OSError, EOFError):
        pass
    return rows

def _as_path_list(maybe: Optional[Union[str, Path]]) -> List[Path]:
     """Accept file/dir/glob/comma-separated specs -> list of existing files."""
     if not maybe:
         return []
     p = Path(maybe)
     if p.exists() and p.is_dir():
         files = sorted(list(p.glob("*.jsonl")) + list(p.glob("*.jsonl.gz")) + list(p.glob("*.gz")))
         return [f for f in files if f.is_file()]
     parts = split_csv_or_globs(str(maybe))
     out: List[Path] = []
     for part in parts:
         pp = Path(part)
         if pp.exists() and pp.is_file():
             out.append(pp)
         else:
             for m in glob.glob(part):
                 out.append(Path(m))
     return sorted(list(dict.fromkeys(out)))
 
def load_softlabel_map(softlabel_files: List[Path], max_lines: Optional[int]=None) -> Dict[str, Dict[str, Any]]:
    """Load teacher soft-labels from JSONL/JSONL.GZ into {id: {topk, steps}}."""
    m: Dict[str, Dict[str, Any]] = {}
    for p in softlabel_files:
        for obj in read_jsonl_any(p, max_lines=max_lines):
            rid = obj.get("id")
            steps = obj.get("steps")
            if rid is None or steps is None:
                continue
            m[str(rid)] = {"topk": obj.get("topk"), "steps": steps}
    return m

def discover_softlabel_files(distilled_jsonls: List[Path]) -> List[Path]:
    """
    Best-effort discovery of softlabel sidecars:
      1) distilled/manifest.json -> datasets[*].softlabels
      2) sibling ../softlabels/*.gz (common layout)
    """
    files: List[Path] = []
    # (1) manifest.json next to distilled outputs
    for dj in distilled_jsonls:
        man = dj.parent / "manifest.json"
        if man.exists():
            try:
                payload = json.loads(man.read_text(encoding="utf-8"))
                for d in payload.get("datasets", []):
                    sp = d.get("softlabels")
                    if sp:
                        p = Path(sp)
                        if p.exists():
                            files.append(p)
            except Exception:
                pass
    # (2) sibling softlabels folder
    for dj in distilled_jsonls:
        if dj.parent.name == "distilled":
            sib = dj.parent.parent / "softlabels"
            if sib.exists() and sib.is_dir():
                files += list(sib.glob("*.jsonl")) + list(sib.glob("*.jsonl.gz")) + list(sib.glob("*.gz"))
    return sorted(list(dict.fromkeys([p for p in files if p.exists() and p.is_file()])))

def glob_many(patterns: List[str]) -> List[Path]:
    out = []
    for p in patterns:
        for m in glob.glob(p):
            out.append(Path(m))
    # stable order
    return sorted(list(dict.fromkeys(out)))


def split_csv_or_globs(s: str) -> List[str]:
    items = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk: continue
        items.append(chunk)
    return items


def sniff_csv_dialect(path: Union[str, Path], sample_bytes: int = 4096):
    """Best-effort delimiter sniffing; fallback to comma."""
    try:
        with open(path, "rb") as f:
            raw = f.read(sample_bytes)
        try:
            return csv.Sniffer().sniff(raw.decode("utf-8", errors="ignore"))
        except Exception:
            class _D: delimiter = ","
            return _D()
    except Exception:
        class _D: delimiter = ","
        return _D()


def csv_to_jsonl_structured(
    csv_path: Path,
    out_dir: Path,
    q_keys: List[str],
    a_keys: List[str],
    c_keys: List[str],
    id_keys: List[str],
) -> Path:
    """
    Convert a CSV/TSV into JSONL with a single structured schema:
      {
        "id": "...",
        "source": "<original csv path>",
        "task": "raw_csv",
        "context": "<optional>",
        "sft": {"input": "<question>", "output": "<gold/answer if present>"}
      }
    We intentionally omit 'instruction' here; the distiller will inject the default
    clinical instruction via extract_fields_any().
    """
    conv_dir = ensure_dir(Path(out_dir) / "converted")
    out_path = conv_dir / f"{csv_path.stem}.jsonl"
    dialect = sniff_csv_dialect(csv_path)
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin, dialect=dialect)
        for i, row in enumerate(reader):
            # Pick fields using the same key detection as JSONL flow
            rid = pick_first(row, id_keys) or f"{csv_path.stem}:{i}"
            q   = pick_first(row, q_keys) or ""
            a   = pick_first(row, a_keys)
            cx  = pick_first(row, c_keys)
            rec: Dict[str, Any] = {
                "id": rid,
                "source": str(csv_path),
                "task": "raw_csv",
                "sft": {"input": q}
            }
            if a:
                rec["sft"]["output"] = a
            if cx:
                # Keep context at top-level so extractor can find it
                rec["context"] = cx
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out_path


# -------- dataset field detection --------
DEFAULT_Q = ["question","Question","query","prompt","user_question","patient","input"]
DEFAULT_A = ["answer","Answer","doctor_answer","response","Response","output","gold","assistant"]
DEFAULT_C = ["context","Context","history","background","case","notes"]
DEFAULT_ID= ["id","ID","_id","uid","q_id","qid"]


def pick_first(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None:
            v = d[k]
            if isinstance(v, (str,int)): return str(v)
    return None


def extract_fields_flat(obj: Dict[str, Any], qk, ak, ck, ik):
    q  = pick_first(obj, qk); a = pick_first(obj, ak)
    cx = pick_first(obj, ck); _id= pick_first(obj, ik)
    return q, a, cx, _id


def extract_fields_any(obj, qk, ak, ck, ik, default_instruction: str):
    rid = obj.get("id")
    sft = obj.get("sft")
    if isinstance(sft, dict) and ("input" in sft or "instruction" in sft or "output" in sft):
        instruction = sft.get("instruction") or default_instruction
        input_text  = sft.get("input") or ""
        context     = obj.get("context") or pick_first(obj, ck)
        gold        = sft.get("output")
        if rid is None: rid = pick_first(obj, DEFAULT_ID)
        return instruction, input_text, context, gold, (rid if rid is not None else None)
    # flat
    q,a,cx,r2 = extract_fields_flat(obj, qk, ak, ck, ik)
    instruction = default_instruction
    input_text  = q or ""
    rid_final   = rid if rid is not None else r2
    return instruction, input_text, cx, a, (rid_final if rid_final is not None else None)


# -------- prompt shaping --------
CLINICAL_CONCISE_INSTRUCTION = (
    "Provide a clinically safe, concise answer within a strict token budget. "
    "Preserve key facts only: brief assessment, differentials (if relevant), "
    "red-flag warnings, and clear next steps. Avoid repetition or filler."
)


def build_input_block(q: str, ctx: Optional[str]) -> str:
    q = (q or "").strip()
    s = f"Question: {q}" if q else "Question:"
    if ctx and ctx.strip(): s += f"\nContext:\n{ctx.strip()}"
    return s


def build_prompt(instruction: str, input_block: str) -> str:
    return f"### Instruction:\n{instruction.strip()}\n\n### Input:\n{input_block.strip()}\n\n### Response:\n"


# -------- teacher loader & gen --------
def _ensure_pad_eos(tok, model):
    added=False
    if tok.eos_token_id is None:
        tok.add_special_tokens({'eos_token':'</s>'}); added=True
    if tok.pad_token_id is None:
        tok.add_special_tokens({'pad_token':'<|pad|>'}); added=True
    if added and hasattr(model,"resize_token_embeddings"):
        model.resize_token_embeddings(len(tok))
    model.config.pad_token_id = tok.pad_token_id


def _ctx_len(model):
    return int(getattr(model.config, "max_position_embeddings",
           getattr(model.config, "n_positions", 4096)))


def _max_input_len(model, max_new_tokens: int, safety_margin: int=8) -> int:
    return max(128, _ctx_len(model) - int(max_new_tokens) - safety_margin)


def load_teacher(teacher_dir: Path, force_slow=False, trust_remote_code=False):
    tok=None; err=None
    if not force_slow:
        try:
            tok = AutoTokenizer.from_pretrained(str(teacher_dir), use_fast=True, trust_remote_code=trust_remote_code)
        except Exception as e:
            err=e
    if tok is None:
        tok = AutoTokenizer.from_pretrained(str(teacher_dir), use_fast=False, trust_remote_code=trust_remote_code)
    tok.padding_side="left"; tok.truncation_side="left"
    model = AutoModelForCausalLM.from_pretrained(
        str(teacher_dir), device_map="auto", trust_remote_code=trust_remote_code,
        torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    _ensure_pad_eos(tok, model)
    if err: print("[warn] fast tokenizer failed; using slow:", repr(err))
    return tok, model


@torch.no_grad()
def generate_texts(tok, model, prompts: List[str], max_new_tokens: int, temperature: float):
    max_inp=_max_input_len(model, max_new_tokens)
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_inp).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature>0.0),
        temperature=(temperature if temperature>0.0 else None),
        use_cache=True,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    full = tok.batch_decode(out, skip_special_tokens=True)
    pre  = tok.batch_decode(enc.input_ids, skip_special_tokens=True)
    return [f[len(p):].strip() for f,p in zip(full, pre)]


@torch.no_grad()
def generate_with_scores(tok, model, prompts: List[str], max_new_tokens: int, temperature: float):
    max_inp=_max_input_len(model, max_new_tokens)
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_inp).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature>0.0),
        temperature=(temperature if temperature>0.0 else None),
        use_cache=True,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )
    seq = out.sequences
    prompt_len = enc.attention_mask.sum(dim=1).tolist()
    gen_ids_list = [s[p:] for s,p in zip(seq, prompt_len)]
    full = tok.batch_decode(seq, skip_special_tokens=True)
    pre  = tok.batch_decode(enc.input_ids, skip_special_tokens=True)
    gen_txt = [f[len(p):].strip() for f,p in zip(full, pre)]
    scores_per_b = [[score[b] for score in out.scores] for b in range(seq.shape[0])]
    out_list=[]
    for t,gids,s in zip(gen_txt, gen_ids_list, scores_per_b):
        out_list.append({"generated_text": t, "generated_ids": gids.detach().cpu(), "scores": s})
    return out_list


def topk_logprobs_per_step(scores_list: List[torch.Tensor], gen_ids: torch.Tensor, k: int):
    if k<=0: return []
    out=[]; logsm=torch.nn.LogSoftmax(dim=-1)
    for t,logits in enumerate(scores_list):
        if logits is None: continue
        logits = logits.detach().float().cpu()
        if logits.numel()==0: continue
        if logits.dim()==2: logits = logits.squeeze(0)
        if logits.dim()!=1: continue
        lp = logsm(logits)
        vocab = lp.shape[0]
        kk=min(k,vocab)
        topv, topi = torch.topk(lp, kk)
        chosen = int(gen_ids[t].item()) if t < gen_ids.numel() else None
        out.append({
            "t": t, "chosen_id": chosen,
            "topk_ids": [int(i) for i in topi.tolist()],
            "topk_logprobs": [float(v) for v in topv.tolist()]
        })
    return out


# -------- finetune dataset & KD --------
from dataclasses import dataclass
from torch.utils.data import Dataset
import torch.nn.functional as F


def prompt_for_row(instr: str, inp: str) -> str:
    return build_prompt(instr, inp)


@dataclass
class DistillDataset(Dataset):
    data: List[Dict[str, Any]]
    tokenizer: Any
    soft_map: Dict[str, Any]
    use_soft: bool
    kd_temp: float
    max_len: int = 2048
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        ex = self.data[idx]
        rid = str(ex.get("id", idx))
        prm = prompt_for_row(ex["instruction"], ex["input"])
        ans = (ex["output"] or "").strip()
        if self.tokenizer.eos_token and not ans.endswith(self.tokenizer.eos_token):
            ans += self.tokenizer.eos_token
        p_ids = self.tokenizer(prm, add_special_tokens=False)["input_ids"]
        a_ids = self.tokenizer(ans, add_special_tokens=False)["input_ids"]
        input_ids = (p_ids + a_ids)[:self.max_len]
        used_prompt_len = min(len(p_ids), len(input_ids))
        labels = [-100]*used_prompt_len
        remain = self.max_len - len(labels)
        labels += a_ids[:max(0, remain)]
        labels = labels[:len(input_ids)]
        attn = [1]*len(input_ids)
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_len": used_prompt_len,
            "id": rid,
        }
        if self.use_soft:
             # Prefer embedded soft_labels; otherwise fall back to sidecar map by id
             soft = ex.get("soft_labels")
             if soft is None and isinstance(self.soft_map, dict):
                 soft = self.soft_map.get(rid)
             item["soft"] = soft
        else:
             item["soft"] = None
        return item


def pad_to_max(batch, pad_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    out = {"input_ids":[], "attention_mask":[], "labels":[], "prompt_len":[], "id":[], "soft":[]}
    for x in batch:
        n=len(x["input_ids"]); pad=max_len-n
        out["input_ids"].append(torch.cat([x["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
        out["attention_mask"].append(torch.cat([x["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        out["labels"].append(torch.cat([x["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        out["prompt_len"].append(x["prompt_len"]); out["id"].append(x["id"]); out["soft"].append(x["soft"])
    return {
        "input_ids": torch.stack(out["input_ids"], dim=0),
        "attention_mask": torch.stack(out["attention_mask"], dim=0),
        "labels": torch.stack(out["labels"], dim=0),
        "prompt_len": torch.tensor(out["prompt_len"], dtype=torch.long),
        "id": out["id"], "soft": out["soft"]
    }


def _steps_to_probs(soft: Dict[str, Any], eps: float=1e-8):
    out=[]
    for st in (soft.get("steps") or []):
        ids = st.get("topk_ids") or []
        lps = st.get("topk_logprobs") or []
        if not ids or not lps:
            out.append(([],[])); continue
        ps = np.exp(np.array(lps, dtype=np.float64))
        s = float(ps.sum()) + eps
        out.append((ids, (ps/s).tolist()))
    return out


def kd_loss_subset(logits: torch.Tensor, prompt_len: torch.Tensor, soft_batch, temperature: float=1.0):
    B,T,V = logits.shape
    dev = logits.device
    total = torch.tensor(0.0, device=dev); cnt=0
    for b in range(B):
        soft = soft_batch[b]
        if not isinstance(soft, dict) or not soft.get("steps"): continue
        pairs = _steps_to_probs(soft)
        start = int(prompt_len[b].item())
        max_j = min(len(pairs), T - start)
        for j in range(max_j):
            ids, probs = pairs[j]
            if not ids: continue
            ids_np=np.asarray(ids, dtype=np.int64)
            probs_np=np.asarray(probs, dtype=np.float64)
            mask = (ids_np>=0) & (ids_np<V)
            if not mask.any(): continue
            ids_f = ids_np[mask]; probs_f=probs_np[mask]
            s = float(probs_f.sum())
            if s <= 0:
                continue
            probs_f = probs_f/s
            ids_t = torch.tensor(ids_f.tolist(), dtype=torch.long, device=dev)
            t_probs = torch.tensor(probs_f.tolist(), dtype=torch.float32, device=dev)
            s_logits_sel = logits[b, start+j, ids_t] / max(1e-6, temperature)
            s_logsumexp = torch.logsumexp(s_logits_sel, dim=-1, keepdim=True)
            s_logp_sel = s_logits_sel - s_logsumexp
            t_logp = torch.log(t_probs + 1e-8)
            kl = torch.sum(t_probs * (t_logp - s_logp_sel))
            total = total + kl; cnt+=1
    return total / max(1,cnt)


class KDTrainer(Trainer):
    def __init__(self, *args, kd_weight: float=0.5, kd_temperature: float=1.0, use_soft_labels: bool=False, **kw):
        super().__init__(*args, **kw)
        self.kd_weight = kd_weight; self.kd_temperature=kd_temperature; self.use_soft_labels=use_soft_labels
    def compute_loss(self, model, inputs, return_outputs=False):
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=inputs["labels"])
        ce = out.loss
        if not self.use_soft_labels:
            return (ce, out) if return_outputs else ce
        kd = kd_loss_subset(out.logits, inputs["prompt_len"], inputs["soft"], self.kd_temperature)
        loss = ce + (self.kd_weight * kd)
        return (loss, out) if return_outputs else loss


# -------- high-level: DISTILL --------
def cmd_distill(args):
    out_dir = ensure_dir(args.out_dir)
    log = setup_file_logger(out_dir / "distill.log")
    dst_dir = ensure_dir(out_dir/"distilled")
    soft_dir= ensure_dir(out_dir/"softlabels") if (args.command=="soft" and args.topk_logprobs>0) else None
    # expand datasets
    patterns = split_csv_or_globs(args.data_files)
    data_paths = glob_many(patterns)
    if not data_paths:
        raise SystemExit("No datasets matched from --data-files")
    # teacher
    tok, model = load_teacher(Path(args.teacher_dir), force_slow=args.force_slow_tokenizer, trust_remote_code=args.trust_remote_code)
    # key overrides
    qk = DEFAULT_Q if not args.question_keys else [s.strip() for s in args.question_keys.split(",") if s.strip()]
    ak = DEFAULT_A if not args.answer_keys   else [s.strip() for s in args.answer_keys.split(",") if s.strip()]
    ck = DEFAULT_C if not args.context_keys  else [s.strip() for s in args.context_keys.split(",") if s.strip()]
    ik = DEFAULT_ID if not args.id_keys      else [s.strip() for s in args.id_keys.split(",") if s.strip()]
    manifest = {"datasets": [], "mode": args.command, "topk": args.topk_logprobs, "max_new_tokens": args.max_new_tokens}
    merged_out = []
    # for each dataset
    for path in data_paths:
        # resolve CSV/TSV → JSONL conversion
        used_path = path
        converted = False
        if path.suffix.lower() in (".csv", ".tsv"):
            used_path = csv_to_jsonl_structured(
                csv_path=path,
                out_dir=out_dir,
                q_keys=qk, a_keys=ak, c_keys=ck, id_keys=ik,
            )
            converted = True
        # now all has been converted to structured JSONL
        rows = list(read_jsonl(used_path))
        stem = path.stem # keep original name for artifacts
        out_jsonl = dst_dir / f"{stem}_distilled.jsonl"
        soft_path = (soft_dir / f"{stem}_softlabels.jsonl.gz") if soft_dir else None
        # if emit soft labels
        n_emit=0; n_skip=0
        soft_f = gzip.open(soft_path, "wt", encoding="utf-8") if soft_path else None
        # batcher
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start+args.batch_size]
            prompts=[]; instrs=[]; inputs=[]; golds=[]; rids=[]
            for i,obj in enumerate(batch):
                instr, inp, ctx, gold, rid = extract_fields_any(obj, qk, ak, ck, ik, args.instruction)
                if not rid: rid = f"{stem}:{start+i}"
                if not inp or not inp.strip():
                    n_skip+=1; continue
                input_block = build_input_block(inp, ctx)
                pr = build_prompt(instr or args.instruction, input_block)
                prompts.append(pr); instrs.append(instr); inputs.append(input_block); golds.append(gold); rids.append(rid)
            if not prompts: continue
            # hard/soft builder
            try:
                if args.command == "text":
                    gens = generate_texts(tok, model, prompts, args.max_new_tokens, args.temperature)

                    for gen, ins, inp, rid, gold in zip(gens, instrs, inputs, rids, golds):
                        rec = {
                            "instruction": ins or args.instruction,
                            "input": inp,
                            "output": gen,
                            "source": f"{stem}_distilled_{Path(args.teacher_dir).name}",
                            "id": rid,
                            "task": "distillation",
                            "meta": {
                                "gen": {
                                    "max_new_tokens": args.max_new_tokens,
                                    "temperature": args.temperature,
                                    "do_sample": (args.temperature > 0.0),
                                }
                            },
                        }

                        if args.include_gold and gold is not None:
                            rec["meta"]["gold"] = gold

                        # append safely
                        append_jsonl_line(out_jsonl, rec)
                        merged_out.append(rec)
                        n_emit += 1

                else:
                    outs = generate_with_scores(tok, model, prompts, args.max_new_tokens, args.temperature)
                    for o,ins,inp,rid,gold in zip(outs, instrs, inputs, rids, golds):
                        rec = {
                            "instruction": ins or args.instruction,
                            "input": inp, "output": o["generated_text"],
                            "source": f"{stem}_distilled_{Path(args.teacher_dir).name}",
                            "id": rid, "task": "distillation",
                            "meta": {"gen":{"max_new_tokens":args.max_new_tokens,"temperature":args.temperature,"do_sample":(args.temperature>0.0)},
                                     "gen_token_count": int(o["generated_ids"].numel())}
                        }
                        if args.include_gold and gold is not None: rec["meta"]["gold"]=gold
                        if args.topk_logprobs>0:
                            soft = topk_logprobs_per_step(o["scores"], o["generated_ids"], args.topk_logprobs)
                            if soft_f:
                                soft_f.write(json.dumps({"id":rid,"topk":args.topk_logprobs,"steps":soft}, ensure_ascii=False)+"\n")
                                # store an audit pointer for downstream auto-discovery
                                rec["soft_labels_ref"] = str(soft_path)
                                rec["soft_labels_topk"] = int(args.topk_logprobs)
                            else:
                                # if no sidecar, embed soft labels directly
                                rec["soft_labels"]={"topk":args.topk_logprobs,"steps":soft}
                        append_jsonl_line(out_jsonl, rec)
                        merged_out.append(rec); n_emit+=1
            except Exception as e:
                log.error(f"[err] dataset={stem} batch_start={start} error={repr(e)}\n{traceback.format_exc()}")
                n_skip += len(prompts)
        # Append metadata
        if soft_f: soft_f.close()
        manifest["datasets"].append({
            "source_path": str(path),
            "used_path": str(used_path),
            "converted_from_csv": bool(converted),
            "distilled": str(out_jsonl),
            "softlabels": (str(soft_path) if soft_path else None),
            "emitted": n_emit,
            "skipped": n_skip
        })
        log.info(f"[{stem}] emitted={n_emit} skipped={n_skip}")
    # merged export
    merged_path = dst_dir / "merged_distilled.jsonl"
    write_jsonl(merged_out, merged_path)
    manifest["merged"] = str(merged_path)
    with open(dst_dir/"manifest.json","w",encoding="utf-8") as f: json.dump(manifest, f, indent=2)
    log.info(f"Distillation complete. Merged: {merged_path}")


# -------- high-level: FINETUNE --------
def train_val_test_split(rows, train_ratio, val_ratio, test_ratio, seed):
    assert abs(train_ratio+val_ratio+test_ratio-1.0) < 1e-6
    rng = random.Random(seed); idx = list(range(len(rows))); rng.shuffle(idx)
    n=len(rows); ntr=int(n*train_ratio); nva=int(n*val_ratio)
    tr=idx[:ntr]; va=idx[ntr:ntr+nva]; te=idx[ntr+nva:]
    return [rows[i] for i in tr], [rows[i] for i in va], [rows[i] for i in te]


def load_distilled_many(paths: List[Path]) -> List[Dict[str, Any]]:
    rows=[]
    for p in paths:
        for obj in read_jsonl(p):
            if all(k in obj for k in ("instruction","input","output")):
                if "id" not in obj: obj["id"] = f"{p.stem}:{len(rows)}"
                rows.append(obj)
    return rows


def attach_lora_or_resume(base_model, resume_adapter_dir, lora_r, lora_alpha, lora_dropout, target_modules):
    if resume_adapter_dir:
        return PeftModel.from_pretrained(base_model, resume_adapter_dir, is_trainable=True)
    lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                          task_type="CAUSAL_LM", target_modules=target_modules)
    return get_peft_model(base_model, lora_cfg)


def cmd_finetune(args):
    out_dir = ensure_dir(args.out_dir)
    log = setup_file_logger(out_dir/"train.log")
    # collect distilled files
    paths=[]
    if args.data_json:
        paths += [Path(args.data_json)]
    if args.data_json_multi:
        paths += glob_many(split_csv_or_globs(args.data_json_multi))
    if args.data_json_dir:
        paths += sorted(Path(args.data_json_dir).glob("*.jsonl"))
    if not paths: raise SystemExit("Provide distilled data with --data-json or --data-json-dir or --data-json-multi")
    # loader
    all_rows = load_distilled_many(paths)
    log.info(f"Loaded {len(all_rows)} distilled rows from {len(paths)} file(s).")
    # split or not
    if args.no_split:
        train_rows, val_rows, test_rows = all_rows, [], []
        log.info("No split: using all rows for training.")
    else:
        train_rows, val_rows, test_rows = train_val_test_split(all_rows, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
        if args.save_splits:
            sp = ensure_dir(out_dir/"splits")
            write_jsonl(train_rows, sp/"train.jsonl")
            write_jsonl(val_rows,   sp/"val.jsonl")
            write_jsonl(test_rows,  sp/"test.jsonl")
            log.info(f"Saved splits to {sp}")
    # tokenizer / model
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    tok.padding_side="left"
    if tok.pad_token_id is None and tok.eos_token_id is not None: tok.pad_token = tok.eos_token
    load_kwargs={}
    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                     bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16)
        load_kwargs["quantization_config"]=bnb_cfg; load_kwargs["device_map"]="auto"
    else:
        load_kwargs["torch_dtype"]= torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
        load_kwargs["device_map"]="auto"
    base = AutoModelForCausalLM.from_pretrained(args.model_dir, **load_kwargs)
    base.config.pad_token_id = tok.pad_token_id
    if args.gradient_checkpointing: base.gradient_checkpointing_enable()
    if args.use_qlora: base = prepare_model_for_kbit_training(base)
    model = attach_lora_or_resume(base, args.resume_adapter_dir, args.lora_r, args.lora_alpha, args.lora_dropout, args.target_modules)
    model.print_trainable_parameters()
    # datasets
    # KD soft-labels (optional)
    use_soft = bool(getattr(args, "use_soft_labels", False))
    if not use_soft:
        # auto-enable if examples already contain embedded soft labels
        use_soft = any(isinstance(r.get("soft_labels"), dict) for r in train_rows)
 
    soft_map: Dict[str, Dict[str, Any]] = {}
    soft_files: List[Path] = []
    if use_soft:
        soft_files = _as_path_list(getattr(args, "softlabels", None))
        if not soft_files:
            soft_files = discover_softlabel_files(paths)
        soft_map = load_softlabel_map(soft_files)
        log.info(f"Soft-labels: enabled={use_soft} files={len(soft_files)} loaded_ids={len(soft_map)}")
        if len(soft_map) == 0 and not any(isinstance(r.get("soft_labels"), dict) for r in train_rows):
            log.warning("Soft-labels enabled but none were found. KD will behave like CE-only.")
 
    train_ds = DistillDataset(train_rows, tok, soft_map, use_soft, args.kd_temperature, max_len=args.max_len)
    val_ds   = DistillDataset(val_rows,   tok, soft_map, use_soft, args.kd_temperature, max_len=args.max_len) if val_rows else None

    def collate(b): return pad_to_max(b, tok.pad_token_id)
    # training args
    ta = {
        "output_dir": str(out_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": getattr(args, "warmup_ratio", 0.0),
        "lr_scheduler_type": getattr(args, "scheduler", "linear"),
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "seed": args.seed,
        "report_to": "none",
        "dataloader_pin_memory": False,
        "remove_unused_columns": False,
        "save_total_limit": 2
    }
    if args.bf16: ta["bf16"]=True
    if args.fp16 and not args.bf16: ta["fp16"]=True
    if val_ds:
        ta["evaluation_strategy"]="steps"
        ta["eval_steps"]=args.eval_steps
        ta["load_best_model_at_end"]=True
        ta["metric_for_best_model"]="eval_loss"
        ta["greater_is_better"]=False
        ta["save_strategy"]="steps"
    # Configs trainer
    if use_soft:
        trainer = KDTrainer(
            model=model,
            args=TrainingArguments(**ta),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate,
            callbacks=([EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if val_ds else []),
            kd_weight=getattr(args, "kd_weight", 0.5),
            kd_temperature=getattr(args, "kd_temperature", 1.0),
            use_soft_labels=True,
        )
    else:
        trainer = Trainer(
            model=model,
            args=TrainingArguments(**ta),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate,
            callbacks=([EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if val_ds else []),
        )
    # Export trainer
    train_out = trainer.train(resume_from_checkpoint=args.resume_trainer_dir)
    log.info(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    # save final
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    # pointer to best
    if trainer.state.best_model_checkpoint:
        best_ptr = out_dir/"BEST_CHECKPOINT.txt"
        best_ptr.write_text(trainer.state.best_model_checkpoint)
        log.info(f"Best checkpoint pointer -> {best_ptr}")
    # export trainer log history
    with open(out_dir/"training_log.json","w",encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2, default=str)
    log.info("Finetune done.")


# -------- glue: ALL --------
def cmd_all(args):
    # distill first
    dargs = argparse.Namespace(**{k:v for k,v in vars(args).items() if k in {
        "command","data_files","teacher_dir","out_dir","instruction","max_new_tokens","temperature","batch_size",
        "force_slow_tokenizer","trust_remote_code","topk_logprobs","include_gold",
        "question_keys","answer_keys","context_keys","id_keys"
    }})
    cmd_distill(dargs)
    # then finetune on merged
    merged = Path(args.out_dir)/"distilled"/"merged_distilled.jsonl"
    fargs = argparse.Namespace(**{
        "model_dir": args.model_dir, "data_json": str(merged), "data_json_multi": None, "data_json_dir": None,
        "out_dir": str(Path(args.out_dir)/"checkpoints"),
        "train_ratio": args.train_ratio, "val_ratio": args.val_ratio, "test_ratio": args.test_ratio,
        "seed": args.seed, "save_splits": args.save_splits, "no_split": args.no_split,
        "use_qlora": args.use_qlora, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout, "target_modules": args.target_modules,
        "bf16": args.bf16, "fp16": args.fp16, "gradient_checkpointing": args.gradient_checkpointing,
        "early_stopping_patience": args.early_stopping_patience,
        "epochs": args.epochs, "batch_size": args.batch_size_ft, "grad_accum": args.grad_accum,
        "lr": args.lr, "weight_decay": args.weight_decay, "warmup_ratio": 0.03,
        "scheduler": "linear", "eval_steps": args.eval_steps, "save_steps": args.save_steps,
        "logging_steps": args.logging_steps, "max_len": args.max_len,
        "resume_adapter_dir": args.resume_adapter_dir, "resume_trainer_dir": args.resume_trainer_dir,
        "use_soft_labels": (args.command=="soft") or getattr(args, "use_soft_labels", False),
        "kd_weight": getattr(args, "kd_weight", 0.5),
        "kd_temperature": getattr(args, "kd_temperature", 1.0),
        "softlabels": (str(Path(args.out_dir)/"softlabels") if (args.command=="soft") else getattr(args, "softlabels", None))
    })
    cmd_finetune(fargs)


# -------- CLI --------
def build_cli():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="subcmd", required=True)
    # shared distill args
    common_dist = {
        "command": ("--command", dict(choices=["text","soft"], default="text")),
        "data_files": ("--data-files", dict(required=True, help="Comma-separated paths/globs, e.g. 'data/*.jsonl,data/other.jsonl'")),
        "teacher_dir": ("--teacher-dir", dict(required=True)),
        "out_dir": ("--out-dir", dict(required=True)),
        "instruction": ("--instruction", dict(default=CLINICAL_CONCISE_INSTRUCTION)),
        "max_new_tokens": ("--max-new-tokens", dict(type=int, default=1024)),
        "temperature": ("--temperature", dict(type=float, default=0.0)),
        "batch_size": ("--batch-size", dict(type=int, default=4)),
        "force_slow_tokenizer": ("--force-slow-tokenizer", dict(action="store_true")),
        "trust_remote_code": ("--trust-remote-code", dict(action="store_true")),
        "topk_logprobs": ("--topk-logprobs", dict(type=int, default=10)),
        "include_gold": ("--include-gold", dict(action="store_true")),
        "question_keys": ("--question-keys", dict(default=None)),
        "answer_keys": ("--answer-keys", dict(default=None)),
        "context_keys": ("--context-keys", dict(default=None)),
        "id_keys": ("--id-keys", dict(default=None)),
    }
    # CLI args
    pd = sub.add_parser("distill", help="Run teacher distillation on 1..N datasets")
    for k,(flag,kw) in common_dist.items(): pd.add_argument(flag, **kw)
    pd.set_defaults(func=cmd_distill)
    # finetune
    pf = sub.add_parser("finetune", help="Finetune LoRA/QLoRA on distilled data")
    pf.add_argument("--model-dir", required=True)
    pf.add_argument("--data-json", default=None, help="One merged JSONL")
    pf.add_argument("--data-json-multi", default=None, help="Comma-separated paths/globs")
    pf.add_argument("--data-json-dir", default=None, help="Directory of .jsonl files")
    pf.add_argument("--out-dir", required=True)
    pf.add_argument("--train-ratio", type=float, default=0.90)
    pf.add_argument("--val-ratio", type=float, default=0.05)
    pf.add_argument("--test-ratio", type=float, default=0.05)
    pf.add_argument("--seed", type=int, default=42)
    pf.add_argument("--save-splits", action="store_true")
    pf.add_argument("--no-split", action="store_true")

    pf.add_argument("--use-qlora", action="store_true")
    pf.add_argument("--lora-r", type=int, default=16)
    pf.add_argument("--lora-alpha", type=int, default=32)
    pf.add_argument("--lora-dropout", type=float, default=0.05)
    pf.add_argument("--target-modules", nargs="+", default=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    pf.add_argument("--bf16", action="store_true"); pf.add_argument("--fp16", action="store_true")
    pf.add_argument("--gradient-checkpointing", action="store_true")

    pf.add_argument("--early-stopping-patience", type=int, default=3)
    pf.add_argument("--epochs", type=int, default=3)
    pf.add_argument("--batch-size", type=int, default=1)
    pf.add_argument("--grad-accum", type=int, default=16)
    pf.add_argument("--lr", type=float, default=2e-4)
    pf.add_argument("--weight-decay", type=float, default=0.0)
    pf.add_argument("--eval-steps", type=int, default=200)
    pf.add_argument("--save-steps", type=int, default=200)
    pf.add_argument("--logging-steps", type=int, default=50)
    pf.add_argument("--max-len", type=int, default=2048)
    pf.add_argument("--resume-adapter-dir", default=None)
    pf.add_argument("--resume-trainer-dir", default=None)
    pf.add_argument("--use-soft-labels", action="store_true",
                    help="Enable KD training using teacher soft-labels (requires embedded soft_labels or softlabels sidecar files).")
    pf.add_argument("--kd-weight", type=float, default=0.5, help="Weight for KD loss added to CE loss.")
    pf.add_argument("--kd-temperature", type=float, default=1.0, help="Temperature for KD loss.")
    pf.add_argument("--softlabels", default=None,
                    help="Optional: file/dir/glob(s) for softlabels JSONL(.gz). If omitted, auto-discovered from manifest or ../softlabels.")
    pf.set_defaults(func=cmd_finetune)
    # all
    pa = sub.add_parser("all", help="Distill then finetune")
    # re-use distill args
    for k,(flag,kw) in common_dist.items(): pa.add_argument(flag, **kw)
    # plus finetune essentials
    pa.add_argument("--model-dir", required=True)
    pa.add_argument("--train-ratio", type=float, default=0.90)
    pa.add_argument("--val-ratio", type=float, default=0.05)
    pa.add_argument("--test-ratio", type=float, default=0.05)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--save-splits", action="store_true")
    pa.add_argument("--no-split", action="store_true")
    pa.add_argument("--use-qlora", action="store_true")
    pa.add_argument("--lora-r", type=int, default=16)
    pa.add_argument("--lora-alpha", type=int, default=32)
    pa.add_argument("--lora-dropout", type=float, default=0.05)
    pa.add_argument("--target-modules", nargs="+", default=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    pa.add_argument("--bf16", action="store_true"); pa.add_argument("--fp16", action="store_true")
    pa.add_argument("--gradient-checkpointing", action="store_true")
    pa.add_argument("--early-stopping-patience", type=int, default=3)
    pa.add_argument("--epochs", type=int, default=3)
    pa.add_argument("--batch-size-ft", type=int, default=1)  # avoid name clash with distill batch-size
    pa.add_argument("--grad-accum", type=int, default=16)
    pa.add_argument("--lr", type=float, default=2e-4)
    pa.add_argument("--weight-decay", type=float, default=0.0)
    pa.add_argument("--eval-steps", type=int, default=200)
    pa.add_argument("--save-steps", type=int, default=200)
    pa.add_argument("--logging-steps", type=int, default=50)
    pa.add_argument("--max-len", type=int, default=2048)
    pa.add_argument("--resume-adapter-dir", default=None)
    pa.add_argument("--resume-trainer-dir", default=None)
    pa.set_defaults(func=cmd_all)
    # Final
    return p


def main():
    parser = build_cli()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
