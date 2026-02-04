#!/usr/bin/env python3
"""
Benchmark multiple rerankers on MedQuAD (local JSON/JSONL).

Dataset:
  data/medquad/processed/medquad_cleaned.jsonl
  data/medquad/processed/medquad_clean.json

Key features:
- Benchmark one or many rerankers in a single run (recommended).
- Two modes:
  (A) Random negatives: 1 positive + (N-1) random negatives
  (B) Retriever->reranker: build candidates by embedding retriever + FAISS topK
- Clean outputs:
  outputs/benchmarks/<run_name>/
    run_config.json
    metrics_summary.json
    timings.csv
    candidate_audit_sample.json
    <model_name>/
      metrics.json
      per_query_results.jsonl
      plots/*.png (optional)

Dependencies:
  pip install -U torch transformers sentence-transformers numpy tqdm ujson pyarrow matplotlib
  pip install -U FlagEmbedding  (optional, for decoder-only rerankers)
  pip install -U faiss-cpu      (optional, for retriever mode)
"""

import os, re, gc, time, json, gzip, random, argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# Optional plotting
HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    pass

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, AutoModelForCausalLM

# Optional: Retriever + FAISS
HAS_FAISS = False
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# Optional: SentenceTransformers
HAS_ST = False
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception:
    HAS_ST = False

# Optional: FlagEmbedding decoder-only reranker inference
HAS_FLAG = False
try:
    from FlagEmbedding.inference.reranker.decoder_only.base import BaseLLMReranker
    HAS_FLAG = True
except Exception:
    HAS_FLAG = False


# ------------------------- small utils -------------------------
def slugify(s: str) -> str:
    s = s.strip().replace(os.sep, "_")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:120]

def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def write_csv(path: Path, header: List[str], rows: List[List]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


# ------------------------- dataset parsing -------------------------
Q_KEYS = ["question", "query", "prompt", "instruction", "q"]
A_KEYS = ["answer", "response", "output", "completion", "a"]

def extract_qa(obj: dict) -> Optional[Tuple[str, str, dict]]:
    q = None
    a = None
    for k in Q_KEYS:
        if k in obj and isinstance(obj[k], str) and obj[k].strip():
            q = obj[k].strip()
            break
    for k in A_KEYS:
        if k in obj and isinstance(obj[k], str) and obj[k].strip():
            a = obj[k].strip()
            break
    if not q or not a:
        return None
    meta = {k: v for k, v in obj.items() if k not in set(Q_KEYS + A_KEYS)}
    return q, a, meta

def load_medquad(data_dir: Path, max_examples: Optional[int], seed: int) -> List[dict]:
    jsonl = data_dir / "medquad_cleaned.jsonl"
    jsonf = data_dir / "medquad_clean.json"

    items = []
    if jsonl.exists():
        iterator = read_jsonl(jsonl)
    elif jsonf.exists():
        data = json.loads(jsonf.read_text(encoding="utf-8"))
        iterator = iter(data if isinstance(data, list) else data.get("data", []))
    else:
        raise FileNotFoundError(f"Missing medquad_cleaned.jsonl or medquad_clean.json in {data_dir}")

    for obj in iterator:
        if not isinstance(obj, dict):
            continue
        out = extract_qa(obj)
        if not out:
            continue
        q, a, meta = out
        items.append({"query": q, "answer": a, "meta": meta})

    rnd = random.Random(seed)
    rnd.shuffle(items)
    if max_examples is not None:
        items = items[:max_examples]

    for i, it in enumerate(items):
        it["qid"] = f"mq_{i:07d}"
    return items


# ------------------------- corpus + candidates -------------------------
def build_corpus(examples: List[dict], include_question_in_doc: bool = False) -> Tuple[List[str], List[str]]:
    docs, doc_ids = [], []
    for i, ex in enumerate(examples):
        doc = ex["answer"]
        if include_question_in_doc:
            doc = ex["query"].strip() + "\n\n" + doc
        docs.append(doc)
        doc_ids.append(f"d_{i:07d}")
    return docs, doc_ids

def make_random_candidates(examples, n_docs: int, candidates_per_query: int, seed: int) -> Dict[str, Dict]:
    rnd = random.Random(seed)
    out = {}
    for i, ex in enumerate(examples):
        qid = ex["qid"]
        pos_idx = i
        negs = set()
        while len(negs) < max(candidates_per_query - 1, 0):
            j = rnd.randrange(n_docs)
            if j != pos_idx:
                negs.add(j)
        cand_indices = [pos_idx] + list(negs)
        rnd.shuffle(cand_indices)
        out[qid] = {
            "query": ex["query"],
            "gold_idx": pos_idx,
            "cand_indices": cand_indices,
            "labels": [1 if j == pos_idx else 0 for j in cand_indices],
        }
    return out

def make_retriever_candidates(examples, docs, embedding_model_name: str, topk: int, batch_size: int, seed: int) -> Dict[str, Dict]:
    if not HAS_ST:
        raise RuntimeError("Need sentence-transformers for --use_retriever")
    if not HAS_FAISS:
        raise RuntimeError("Need faiss for --use_retriever")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st = SentenceTransformer(embedding_model_name, device=device)

    # encode corpus
    doc_emb = st.encode(docs, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True).astype("float32")
    dim = doc_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_emb)

    out = {}
    for i, ex in enumerate(tqdm(examples, desc="Retriever candidates")):
        qid = ex["qid"]
        q = ex["query"]
        q_emb = st.encode([q], batch_size=1, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        _, idxs = index.search(q_emb, topk)
        cand_indices = idxs[0].tolist()

        pos_idx = i
        if pos_idx not in cand_indices:
            cand_indices[-1] = pos_idx  # force include positive

        labels = [1 if j == pos_idx else 0 for j in cand_indices]
        out[qid] = {"query": q, "gold_idx": pos_idx, "cand_indices": cand_indices, "labels": labels}
    return out


# ------------------------- metrics -------------------------
def dcg(rels: np.ndarray) -> float:
    denom = np.log2(np.arange(2, rels.size + 2))
    gains = (2 ** rels - 1)
    return float(np.sum(gains / denom))

def ndcg_at_k(rels: List[int], k: int) -> float:
    r = np.array(rels[:k], dtype=np.float32)
    ideal = np.array(sorted(rels, reverse=True)[:k], dtype=np.float32)
    idcg = dcg(ideal)
    return 0.0 if idcg <= 0 else dcg(r) / idcg

def mrr_at_k(rels: List[int], k: int) -> float:
    for i, r in enumerate(rels[:k]):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0

def hit_at_k(rels: List[int], k: int) -> float:
    return 1.0 if any(r > 0 for r in rels[:k]) else 0.0

def ap_at_k(rels: List[int], k: int) -> float:
    hits = 0
    precisions = []
    for i, r in enumerate(rels[:k]):
        if r > 0:
            hits += 1
            precisions.append(hits / (i + 1))
    return float(np.mean(precisions)) if precisions else 0.0

def aggregate_metrics(per_query_rels: List[List[int]], ks: List[int]) -> Dict[str, float]:
    out = {}
    n = len(per_query_rels)
    for k in ks:
        out[f"MRR@{k}"] = float(np.mean([mrr_at_k(r, k) for r in per_query_rels])) if n else 0.0
        out[f"nDCG@{k}"] = float(np.mean([ndcg_at_k(r, k) for r in per_query_rels])) if n else 0.0
        out[f"Recall@{k}"] = float(np.mean([hit_at_k(r, k) for r in per_query_rels])) if n else 0.0
        out[f"MAP@{k}"] = float(np.mean([ap_at_k(r, k) for r in per_query_rels])) if n else 0.0
    return out


# ------------------------- model adapters -------------------------
@dataclass
class ModelResult:
    ranked_idx: List[int]
    ranked_scores: List[float]
    ranked_labels: List[int]

class BaseAdapter:
    def __init__(self, name: str):
        self.name = name
    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        raise NotImplementedError
    def close(self):
        pass

class HFCrossEncoderAdapter(BaseAdapter):
    def __init__(self, name: str, model_path: str, device: str, max_length: int, dtype: str = "fp16"):
        super().__init__(name)
        self.device = device
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

        if dtype == "fp16" and device.startswith("cuda"):
            self.autocast = torch.cuda.amp.autocast(dtype=torch.float16)
        elif dtype == "bf16" and device.startswith("cuda"):
            self.autocast = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            self.autocast = torch.autocast(device_type="cpu", enabled=False)

    @torch.inference_mode()
    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        qs = [p[0] for p in pairs]
        ps = [p[1] for p in pairs]
        enc = self.tok(qs, ps, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.autocast:
            out = self.model(**enc)
            logits = out.logits
            if logits.ndim == 2 and logits.shape[1] > 1:
                scores = logits[:, -1]
            else:
                scores = logits.squeeze(-1)
        return scores.detach().float().cpu().tolist()

    def close(self):
        del self.model, self.tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class FlagDecoderOnlyAdapter(BaseAdapter):
    def __init__(self, name: str, model_path: str, use_fp16: bool, batch_size: int, query_max_len: int, max_len: int, cache_dir: Optional[str] = None):
        super().__init__(name)
        if not HAS_FLAG:
            raise RuntimeError("FlagEmbedding not installed but required for decoder-only rerankers.")
        self.r = BaseLLMReranker(
            model_path,
            use_fp16=use_fp16,
            cache_dir=cache_dir,
            batch_size=batch_size,
            query_max_length=query_max_len,
            max_length=max_len,
            normalize=False,
        )

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        return self.r.compute_score(pairs, batch_size=self.r.batch_size, max_length=self.r.max_length, query_max_length=self.r.query_max_length)

    def close(self):
        del self.r
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class HFPromptLLMAdapter(BaseAdapter):
    def __init__(self, name: str, model_path: str, device: str, max_input_tokens: int = 1024, max_new_tokens: int = 8):
        super().__init__(name)
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

    def _format(self, q: str, p: str) -> str:
        return (
            "Score relevance of Passage to Query for medical retrieval.\n"
            "Return ONLY a number between 0.0 and 1.0.\n\n"
            f"Query:\n{q}\n\nPassage:\n{p}\n\nScore:"
        )

    @torch.inference_mode()
    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        scores = []
        for q, p in pairs:
            prompt = self._format(q, p)
            enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            gen = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=0.0, top_p=1.0,
                eos_token_id=self.tok.eos_token_id,
            )
            text = self.tok.decode(gen[0], skip_special_tokens=True)
            m = re.findall(r"([0-1](?:\.\d+)?)", text.split("Score:")[-1])
            s = float(m[-1]) if m else 0.0
            scores.append(max(0.0, min(1.0, s)))
        return scores

    def close(self):
        del self.model, self.tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def pick_adapter(model_name: str, model_path: str, device: str, args) -> BaseAdapter:
    lname = model_name.lower()
    # Prefer FlagEmbedding for gemma-based rerankers
    if ("gemma" in lname and "reranker" in lname) or ("medswin" in lname and "reranker" in lname):
        if HAS_FLAG:
            return FlagDecoderOnlyAdapter(
                name=model_name,
                model_path=model_path,
                use_fp16=args.fp16,
                batch_size=args.score_batch_size,
                query_max_len=args.query_max_len,
                max_len=args.passage_max_len,
                cache_dir=args.cache_dir,
            )
    # Try seq-cls
    try:
        _ = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        dtype = "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32"
        return HFCrossEncoderAdapter(model_name, model_path, device, args.cross_encoder_max_len, dtype=dtype)
    except Exception:
        # fallback LLM prompt scoring (slow)
        return HFPromptLLMAdapter(model_name, model_path, device, max_input_tokens=min(args.cross_encoder_max_len, 1024), max_new_tokens=8)


# ------------------------- ranking -------------------------
def rank_one_query(adapter: BaseAdapter, q: str, cand_indices: List[int], docs: List[str], labels: List[int], batch_size: int) -> ModelResult:
    pairs = [(q, docs[i]) for i in cand_indices]
    scores = []
    for i in range(0, len(pairs), batch_size):
        scores.extend(adapter.score_pairs(pairs[i:i+batch_size]))
    order = np.argsort(-np.array(scores))
    ranked_idx = [cand_indices[i] for i in order]
    ranked_scores = [float(scores[i]) for i in order]
    ranked_labels = [int(labels[i]) for i in order]
    return ModelResult(ranked_idx, ranked_scores, ranked_labels)

def plot_metrics_bar(metrics: Dict[str, float], out_png: Path, title: str):
    if not HAS_MPL:
        return
    keys = [k for k in metrics.keys() if any(x in k for x in ["MRR@", "nDCG@", "Recall@", "MAP@"])]
    vals = [metrics[k] for k in keys]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(keys)), vals)
    plt.xticks(range(len(keys)), keys, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


# ------------------------- model selection -------------------------
def resolve_models(model_dir: Path, models: List[str]) -> List[Tuple[str, str]]:
    """
    models can be:
      - model names (relative to model_dir)
      - explicit paths (model/... or /abs/path)
    Returns list of (model_name, model_path).
    """
    resolved = []
    for m in models:
        p = Path(m)
        if p.exists():
            resolved.append((p.name, str(p)))
        else:
            p2 = model_dir / m
            if p2.exists():
                resolved.append((m, str(p2)))
            else:
                print(f"[warn] model not found: {m} (looked for {p} and {p2})")
    return resolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/medquad/processed")
    ap.add_argument("--model_dir", type=str, default="model")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Model names or paths. If omitted, uses all subfolders in --model_dir.")
    ap.add_argument("--output_root", type=str, default="outputs/benchmarks")
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--tag", type=str, default=None)

    ap.add_argument("--max_examples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)

    # candidates
    ap.add_argument("--candidates_per_query", type=int, default=64)
    ap.add_argument("--use_retriever", action="store_true")
    ap.add_argument("--retriever_model", type=str, default="sentence-transformers/embeddinggemma-300m-medical")
    ap.add_argument("--retriever_topk", type=int, default=64)
    ap.add_argument("--retriever_bs", type=int, default=64)

    # scoring
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--score_batch_size", type=int, default=32)
    ap.add_argument("--cross_encoder_max_len", type=int, default=512)
    ap.add_argument("--query_max_len", type=int, default=256)
    ap.add_argument("--passage_max_len", type=int, default=1024)
    ap.add_argument("--cache_dir", type=str, default=None)

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")

    ap.add_argument("--ks", type=str, default="1,3,5,10")

    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    # run naming
    mode = "retriever" if args.use_retriever else f"neg{args.candidates_per_query}"
    retr_slug = slugify(Path(args.retriever_model).name if args.use_retriever else "none")
    tag = slugify(args.tag) if args.tag else None
    if args.run_name:
        run_name = slugify(args.run_name)
    else:
        parts = ["medquad", mode, f"N{args.max_examples}", f"seed{args.seed}"]
        if args.use_retriever:
            parts.append(f"retr_{retr_slug}_K{args.retriever_topk}")
        if tag:
            parts.append(tag)
        parts.append(str(int(time.time())))
        run_name = "_".join(parts)

    out_dir = Path(args.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_config.json", vars(args))

    # 1) load dataset
    examples = load_medquad(data_dir, max_examples=args.max_examples, seed=args.seed)
    print(f"[data] loaded {len(examples)} examples from {data_dir}")

    # 2) corpus
    docs, doc_ids = build_corpus(examples, include_question_in_doc=False)

    # 3) candidates
    if args.use_retriever:
        print("[cand] building candidates via retriever ...")
        candidates = make_retriever_candidates(examples, docs, args.retriever_model, args.retriever_topk, args.retriever_bs, args.seed)
    else:
        print("[cand] building candidates via random negatives ...")
        candidates = make_random_candidates(examples, len(docs), args.candidates_per_query, args.seed)

    # audit sample
    audit = []
    for i, ex in enumerate(examples[:10]):
        qid = ex["qid"]
        item = candidates[qid]
        audit.append({
            "qid": qid,
            "query": item["query"],
            "gold_doc_id": doc_ids[item["gold_idx"]],
            "cand_doc_ids": [doc_ids[j] for j in item["cand_indices"][:10]],
            "labels": item["labels"][:10],
        })
    write_json(out_dir / "candidate_audit_sample.json", audit)

    # 4) resolve models
    if args.models is None:
        # all subfolders under model_dir
        models = [p.name for p in sorted(model_dir.iterdir()) if p.is_dir()]
    else:
        models = args.models
    resolved = resolve_models(model_dir, models)
    if not resolved:
        raise RuntimeError("No models resolved. Pass --models <name/path> ... or ensure --model_dir has subfolders.")

    overall_summary = {}
    timings_rows = [["model", "n_queries", "n_pairs", "total_sec", "pairs_per_sec", "p50_query_ms", "p95_query_ms"]]
    qids = [ex["qid"] for ex in examples]

    for model_name, model_path in resolved:
        safe_name = slugify(model_name)
        print(f"\n=== Benchmarking: {model_name} ===")
        adapter = pick_adapter(model_name, model_path, args.device, args)

        per_query_rels = []
        per_query_out = []
        per_query_lat_ms = []
        n_pairs = 0
        t0 = time.time()

        for qid in tqdm(qids, desc=f"Scoring {model_name}"):
            item = candidates[qid]
            cand_indices = item["cand_indices"]
            labels = item["labels"]
            n_pairs += len(cand_indices)

            qt0 = time.time()
            res = rank_one_query(adapter, item["query"], cand_indices, docs, labels, args.score_batch_size)
            qt1 = time.time()
            per_query_lat_ms.append((qt1 - qt0) * 1000.0)

            per_query_rels.append(res.ranked_labels)

            topn = min(10, len(res.ranked_idx))
            per_query_out.append({
                "qid": qid,
                "query": item["query"],
                "gold_doc_id": doc_ids[item["gold_idx"]],
                "top_doc_ids": [doc_ids[j] for j in res.ranked_idx[:topn]],
                "top_scores": res.ranked_scores[:topn],
                "top_labels": res.ranked_labels[:topn],
            })

        total_sec = time.time() - t0
        pairs_per_sec = n_pairs / max(total_sec, 1e-9)
        p50 = float(np.percentile(per_query_lat_ms, 50))
        p95 = float(np.percentile(per_query_lat_ms, 95))

        metrics = aggregate_metrics(per_query_rels, ks=ks)
        metrics.update({
            "n_queries": len(qids),
            "n_pairs": n_pairs,
            "total_seconds": total_sec,
            "pairs_per_second": pairs_per_sec,
            "p50_query_ms": p50,
            "p95_query_ms": p95,
            "model_path": model_path,
        })

        model_out_dir = out_dir / safe_name
        model_out_dir.mkdir(parents=True, exist_ok=True)
        write_json(model_out_dir / "metrics.json", metrics)
        write_jsonl(model_out_dir / "per_query_results.jsonl", per_query_out)

        if HAS_MPL:
            plot_metrics_bar({k: metrics[k] for k in metrics if any(x in k for x in ["MRR@", "nDCG@", "Recall@", "MAP@"])},
                             model_out_dir / "plots" / "metrics_bar.png",
                             title=f"{model_name} metrics")
            plt.figure(figsize=(8, 4))
            plt.hist(per_query_lat_ms, bins=50)
            plt.title(f"{model_name} per-query latency (ms)")
            plt.xlabel("ms"); plt.ylabel("count")
            plt.tight_layout()
            (model_out_dir / "plots").mkdir(parents=True, exist_ok=True)
            plt.savefig(model_out_dir / "plots" / "latency_hist.png")
            plt.close()

        overall_summary[model_name] = metrics
        timings_rows.append([model_name, len(qids), n_pairs, f"{total_sec:.3f}", f"{pairs_per_sec:.2f}", f"{p50:.2f}", f"{p95:.2f}"])

        adapter.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(out_dir / "metrics_summary.json", overall_summary)
    write_csv(out_dir / "timings.csv", timings_rows[0], timings_rows[1:])

    print(f"\n[done] results saved to: {out_dir}")
    print(f"  - {out_dir/'metrics_summary.json'}")
    print(f"  - {out_dir/'timings.csv'}")
    print(f"  - per-model folders inside: {out_dir}")


if __name__ == "__main__":
    main()