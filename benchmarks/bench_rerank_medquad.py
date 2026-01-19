#!/usr/bin/env python3
"""
Reranker benchmarking on MedQuAD (local JSON/JSONL).

Dataset:
  data/medquad/processed/medquad_cleaned.jsonl
  data/medquad/processed/medquad_clean.json

Models (local folders under model/):
  medswin-reranker-bge-gemma
  jina-reranker-v3
  bge-reranker-v2-m3
  bge-reranker-v2-gemma
  qwen3-vl-reranker-8b

What it does:
1) Load MedQuAD QA pairs.
2) Build a "corpus" of passages (default: answers, optionally answer+extra fields if present).
3) Build candidate sets per query:
    - random negatives OR
    - retriever topK via EmbeddingGemma-300M-medical (optional)
4) For each reranker model:
    - score query-passage pairs (batched)
    - rank candidates
    - compute IR metrics: MRR@K, nDCG@K, Recall@K (Hit@K), MAP@K
    - compute speed stats: total time, pairs/sec, p50/p95 per-query latency
5) Save JSON + CSV + (optional) plots.

Dependencies:
  pip install -U torch transformers datasets sentence-transformers numpy tqdm ujson pyarrow matplotlib
  pip install -U "FlagEmbedding"   (only needed for some decoder-only rerankers)

Run from repo root:
  python scripts/bench_rerank_medquad.py --data_dir data/medquad/processed
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

# Torch/Transformers
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, AutoModelForCausalLM

# Optional: Retriever + FAISS
HAS_FAISS = False
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# Optional: SentenceTransformers (EmbeddingGemma-300M-medical is ST-compatible)
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


# ------------------------- IO helpers -------------------------
def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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
    """Best-effort extraction from arbitrary QA-ish JSON shapes."""
    q = None
    a = None

    # direct keys
    for k in Q_KEYS:
        if k in obj and isinstance(obj[k], str) and obj[k].strip():
            q = obj[k].strip()
            break
    for k in A_KEYS:
        if k in obj and isinstance(obj[k], str) and obj[k].strip():
            a = obj[k].strip()
            break

    # nested common patterns
    if q is None:
        for k in ["input", "inputs", "question_text"]:
            if k in obj and isinstance(obj[k], str) and obj[k].strip():
                q = obj[k].strip()
                break
    if a is None:
        for k in ["answer_text", "target", "targets"]:
            if k in obj and isinstance(obj[k], str) and obj[k].strip():
                a = obj[k].strip()
                break

    if not q or not a:
        return None
    meta = {k: v for k, v in obj.items() if k not in set(Q_KEYS + A_KEYS)}
    return q, a, meta

def load_medquad(data_dir: Path, max_examples: Optional[int], seed: int) -> List[dict]:
    """
    Load from JSONL preferred, else JSON.
    Returns list of dicts: {qid, query, answer, meta}
    """
    jsonl = data_dir / "medquad_cleaned.jsonl"
    jsonf = data_dir / "medquad_clean.json"

    items = []
    if jsonl.exists():
        iterator = read_jsonl(jsonl)
    elif jsonf.exists():
        data = json.loads(jsonf.read_text(encoding="utf-8"))
        iterator = iter(data if isinstance(data, list) else data.get("data", []))
    else:
        raise FileNotFoundError(f"Could not find medquad_cleaned.jsonl or medquad_clean.json in {data_dir}")

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

    # assign qids
    for i, it in enumerate(items):
        it["qid"] = f"mq_{i:07d}"
    return items


# ------------------------- candidate generation -------------------------
def build_corpus(examples: List[dict], include_question_in_doc: bool = False) -> Tuple[List[str], List[str]]:
    """
    Corpus documents from answers (optionally include question).
    Returns (doc_texts, doc_ids) aligned by index.
    """
    docs = []
    doc_ids = []
    for i, ex in enumerate(examples):
        doc = ex["answer"]
        if include_question_in_doc:
            doc = ex["query"].strip() + "\n\n" + doc
        docs.append(doc)
        doc_ids.append(f"d_{i:07d}")
    return docs, doc_ids

def make_random_candidates(
    examples: List[dict],
    docs: List[str],
    doc_ids: List[str],
    candidates_per_query: int,
    seed: int
) -> Dict[str, Dict]:
    """
    For each query: 1 positive (its own answer doc) + random negatives from other docs.
    """
    rnd = random.Random(seed)
    n = len(examples)
    assert n == len(docs)

    out = {}
    for i, ex in enumerate(examples):
        qid = ex["qid"]
        pos_idx = i
        # sample negatives
        neg_indices = set()
        while len(neg_indices) < max(candidates_per_query - 1, 0):
            j = rnd.randrange(n)
            if j != pos_idx:
                neg_indices.add(j)
        cand_indices = [pos_idx] + list(neg_indices)
        rnd.shuffle(cand_indices)

        out[qid] = {
            "query": ex["query"],
            "gold_doc_id": doc_ids[pos_idx],
            "cand_doc_ids": [doc_ids[j] for j in cand_indices],
            "cand_docs": [docs[j] for j in cand_indices],
            "labels": [1 if j == pos_idx else 0 for j in cand_indices],
            "meta": ex.get("meta", {}),
        }
    return out

def make_retriever_candidates(
    examples: List[dict],
    docs: List[str],
    doc_ids: List[str],
    embedding_model_name: str,
    topk: int,
    seed: int,
    batch_size: int = 64,
) -> Dict[str, Dict]:
    """
    Retriever -> candidates: encode docs + queries with sentence-transformers model,
    FAISS IP index (cosine via normalization), retrieve topK docs per query, ensure positive included.
    """
    if not HAS_ST:
        raise RuntimeError("sentence-transformers not installed. Install it or use random candidates.")
    if not HAS_FAISS:
        raise RuntimeError("faiss not installed. Install faiss-cpu/faiss-gpu or use random candidates.")

    st = SentenceTransformer(embedding_model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    # encode corpus
    doc_emb = st.encode(docs, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    doc_emb = doc_emb.astype("float32")
    dim = doc_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_emb)

    out = {}
    for i, ex in enumerate(tqdm(examples, desc="Retrieving candidates")):
        qid = ex["qid"]
        q = ex["query"]
        q_emb = st.encode([q], batch_size=1, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, idxs = index.search(q_emb, topk)
        idxs = idxs[0].tolist()

        pos_idx = i
        if pos_idx not in idxs:
            # force include positive (so reranker metrics are meaningful)
            idxs[-1] = pos_idx

        cand_indices = idxs
        out[qid] = {
            "query": q,
            "gold_doc_id": doc_ids[pos_idx],
            "cand_doc_ids": [doc_ids[j] for j in cand_indices],
            "cand_docs": [docs[j] for j in cand_indices],
            "labels": [1 if j == pos_idx else 0 for j in cand_indices],
            "meta": ex.get("meta", {}),
        }
    return out


# ------------------------- metrics -------------------------
def dcg(rels: np.ndarray) -> float:
    # rels are ordered by rank
    denom = np.log2(np.arange(2, rels.size + 2))
    gains = (2 ** rels - 1)
    return float(np.sum(gains / denom))

def ndcg_at_k(rels: List[int], k: int) -> float:
    r = np.array(rels[:k], dtype=np.float32)
    ideal = np.array(sorted(rels, reverse=True)[:k], dtype=np.float32)
    idcg = dcg(ideal)
    if idcg <= 0:
        return 0.0
    return dcg(r) / idcg

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
    ranked_doc_ids: List[str]
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
    """
    For models like bge-reranker-v2-m3, jina-reranker-v3 (if seq-cls), etc.
    """
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
        enc = self.tok(
            qs, ps,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.autocast:
            out = self.model(**enc)
            logits = out.logits
            if logits.ndim == 2 and logits.shape[1] > 1:
                scores = logits[:, -1]  # assume last logit = "relevant"
            else:
                scores = logits.squeeze(-1)
        return scores.detach().float().cpu().tolist()

    def close(self):
        try:
            del self.model
            del self.tok
        except:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class FlagDecoderOnlyAdapter(BaseAdapter):
    """
    For decoder-only rerankers supported by FlagEmbedding BaseLLMReranker
    (e.g., bge-reranker-v2-gemma, medswin-reranker-bge-gemma if compatible).
    """
    def __init__(self, name: str, model_path: str, use_fp16: bool, batch_size: int, query_max_len: int, max_len: int, cache_dir: Optional[str] = None):
        super().__init__(name)
        if not HAS_FLAG:
            raise RuntimeError("FlagEmbedding not installed but required for this adapter.")
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
        try:
            del self.r
        except:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class HFPromptLLMAdapter(BaseAdapter):
    """
    Fallback for LLM-based rerankers that are not supported by FlagEmbedding.
    Scores by prompting the model to output a scalar in [0,1]. Slow but works offline.
    Good for qwen3-vl-reranker-8b *text-only benchmarking*.
    """
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
            "You are a medical information retrieval reranker.\n"
            "Score how relevant the Passage is to the Query.\n"
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
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                eos_token_id=self.tok.eos_token_id,
            )
            text = self.tok.decode(gen[0], skip_special_tokens=True)
            # parse last number
            m = re.findall(r"([0-1](?:\.\d+)?)", text.split("Score:")[-1])
            if m:
                s = float(m[-1])
                s = max(0.0, min(1.0, s))
            else:
                s = 0.0
            scores.append(s)
        return scores

    def close(self):
        try:
            del self.model
            del self.tok
        except:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def pick_adapter(model_name: str, model_path: str, device: str, args) -> BaseAdapter:
    """
    Heuristics:
    - If model name contains 'bge-reranker-v2-gemma' or 'medswin' and FlagEmbedding is available -> FlagDecoderOnlyAdapter
    - Else attempt HF seq classification
    - Else fallback to prompt LLM adapter
    """
    lname = model_name.lower()
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

    # try sequence classification
    try:
        _ = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return HFCrossEncoderAdapter(
            name=model_name,
            model_path=model_path,
            device=device,
            max_length=args.cross_encoder_max_len,
            dtype=("bf16" if args.bf16 else "fp16" if args.fp16 else "fp32"),
        )
    except Exception:
        # fallback LLM prompt scoring
        return HFPromptLLMAdapter(
            name=model_name,
            model_path=model_path,
            device=device,
            max_input_tokens=min(args.cross_encoder_max_len, 1024),
            max_new_tokens=8,
        )


# ------------------------- benchmarking loop -------------------------
def rank_one_query(adapter: BaseAdapter, q: str, doc_ids: List[str], docs: List[str], labels: List[int], batch_size: int) -> ModelResult:
    pairs = [(q, d) for d in docs]
    scores = []
    # batched scoring
    for i in range(0, len(pairs), batch_size):
        scores.extend(adapter.score_pairs(pairs[i:i+batch_size]))
    order = np.argsort(-np.array(scores))
    ranked_doc_ids = [doc_ids[i] for i in order]
    ranked_scores  = [float(scores[i]) for i in order]
    ranked_labels  = [int(labels[i]) for i in order]
    return ModelResult(ranked_doc_ids, ranked_scores, ranked_labels)

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/medquad/processed")
    ap.add_argument("--model_dir", type=str, default="model")
    ap.add_argument("--output_root", type=str, default="outputs/benchmarks")

    ap.add_argument("--max_examples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)

    # candidate settings
    ap.add_argument("--candidates_per_query", type=int, default=64)
    ap.add_argument("--use_retriever", action="store_true", help="Use EmbeddingGemma retriever to build candidates instead of random negatives.")
    ap.add_argument("--retriever_model", type=str, default="sentence-transformers/embeddinggemma-300m-medical")
    ap.add_argument("--retriever_topk", type=int, default=64)
    ap.add_argument("--retriever_bs", type=int, default=64)

    # scoring settings
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--score_batch_size", type=int, default=32, help="Batch size for scoring query-passage pairs.")
    ap.add_argument("--cross_encoder_max_len", type=int, default=512)
    ap.add_argument("--query_max_len", type=int, default=256)
    ap.add_argument("--passage_max_len", type=int, default=1024)
    ap.add_argument("--cache_dir", type=str, default=None)

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")

    # metrics
    ap.add_argument("--ks", type=str, default="1,3,5,10", help="Comma-separated K values for IR metrics.")

    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    rnd = random.Random(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    run_name = f"medquad_{'retriever' if args.use_retriever else 'random'}_{args.max_examples}_{int(time.time())}"
    out_dir = Path(args.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load dataset
    examples = load_medquad(data_dir, max_examples=args.max_examples, seed=args.seed)
    print(f"[data] loaded {len(examples)} examples from {data_dir}")

    # 2) Build corpus
    docs, doc_ids = build_corpus(examples, include_question_in_doc=False)

    # 3) Candidate sets
    if args.use_retriever:
        print("[cand] building candidates via retriever ...")
        candidates = make_retriever_candidates(
            examples, docs, doc_ids,
            embedding_model_name=args.retriever_model,
            topk=args.retriever_topk,
            seed=args.seed,
            batch_size=args.retriever_bs,
        )
    else:
        print("[cand] building candidates via random negatives ...")
        candidates = make_random_candidates(
            examples, docs, doc_ids,
            candidates_per_query=args.candidates_per_query,
            seed=args.seed
        )

    # Save a small audit sample of candidates
    audit_sample = []
    for i, (qid, item) in enumerate(candidates.items()):
        if i >= 10:
            break
        audit_sample.append({
            "qid": qid,
            "query": item["query"],
            "gold_doc_id": item["gold_doc_id"],
            "cand_doc_ids": item["cand_doc_ids"][:10],
            "labels": item["labels"][:10],
        })
    write_json(out_dir / "candidate_audit_sample.json", audit_sample)

    # 4) Benchmark models (auto-discover from your list)
    model_names = [
        "medswin-reranker-bge-gemma",
        "jina-reranker-v3",
        "bge-reranker-v2-m3",
        "bge-reranker-v2-gemma",
        "qwen3-vl-reranker-8b",
    ]
    model_paths = {name: str(model_dir / name) for name in model_names}

    overall_summary = {}
    timings_rows = [["model", "n_queries", "n_pairs", "total_sec", "pairs_per_sec", "p50_query_ms", "p95_query_ms"]]

    # Pre-build query order
    qids = [ex["qid"] for ex in examples]

    for name in model_names:
        path = model_paths[name]
        if not Path(path).exists():
            print(f"[skip] model not found: {path}")
            continue

        print(f"\n=== Benchmarking: {name} ===")
        adapter = pick_adapter(name, path, args.device, args)

        per_query_rels = []
        per_query_out = []
        per_query_lat_ms = []

        n_pairs = 0
        t0 = time.time()

        for qid in tqdm(qids, desc=f"Scoring {name}"):
            item = candidates[qid]
            q = item["query"]
            cand_ids = item["cand_doc_ids"]
            cand_docs = item["cand_docs"]
            labels = item["labels"]

            n_pairs += len(cand_docs)
            qt0 = time.time()
            res = rank_one_query(adapter, q, cand_ids, cand_docs, labels, batch_size=args.score_batch_size)
            qt1 = time.time()
            per_query_lat_ms.append((qt1 - qt0) * 1000.0)

            per_query_rels.append(res.ranked_labels)

            per_query_out.append({
                "qid": qid,
                "query": q,
                "gold_doc_id": item["gold_doc_id"],
                "top_doc_ids": res.ranked_doc_ids[: min(10, len(res.ranked_doc_ids))],
                "top_scores":  res.ranked_scores[: min(10, len(res.ranked_scores))],
                "top_labels":  res.ranked_labels[: min(10, len(res.ranked_labels))],
            })

        t1 = time.time()
        total_sec = t1 - t0
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
        })

        # Save per-model artifacts
        model_out_dir = out_dir / name
        model_out_dir.mkdir(parents=True, exist_ok=True)
        write_json(model_out_dir / "metrics.json", metrics)
        write_jsonl(model_out_dir / "per_query_results.jsonl", per_query_out)

        if HAS_MPL:
            plot_metrics_bar({k: metrics[k] for k in metrics if any(x in k for x in ["MRR@", "nDCG@", "Recall@", "MAP@"])},
                             model_out_dir / "plots" / "metrics_bar.png",
                             title=f"{name} metrics")

            # latency histogram
            plt.figure(figsize=(8,4))
            plt.hist(per_query_lat_ms, bins=50)
            plt.title(f"{name} per-query latency (ms)")
            plt.xlabel("ms")
            plt.ylabel("count")
            plt.tight_layout()
            (model_out_dir / "plots").mkdir(parents=True, exist_ok=True)
            plt.savefig(model_out_dir / "plots" / "latency_hist.png")
            plt.close()

        overall_summary[name] = metrics
        timings_rows.append([name, len(qids), n_pairs, f"{total_sec:.3f}", f"{pairs_per_sec:.2f}", f"{p50:.2f}", f"{p95:.2f}"])

        adapter.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save overall summary
    write_json(out_dir / "metrics_summary.json", overall_summary)
    write_csv(out_dir / "timings.csv", timings_rows[0], timings_rows[1:])

    print(f"\n[done] results saved to: {out_dir}")
    print("Key files:")
    print(f"  - {out_dir / 'metrics_summary.json'}")
    print(f"  - {out_dir / 'timings.csv'}")
    print(f"  - per-model folders: {out_dir / '<model_name>'}")


if __name__ == "__main__":
    main()

