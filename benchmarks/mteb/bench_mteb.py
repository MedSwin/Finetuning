#!/usr/bin/env python
r"""
Windows-ready MTEB reranking benchmark runner for:
  1. MedSwin/MedSwin-Reranker-bge-gemma
  2. BAAI/bge-reranker-v2-gemma
  3. BAAI/bge-reranker-v2-m3

Expected project layout:
  Downloads/mteb/
    script/bench_mteb.py
    model/medswin-reranker-bge-gemma/
    model/bge-reranker-v2-gemma/
    model/bge-reranker-v2-m3/
    outputs/<run-name>/...
    cache/huggingface/...
    cache/mteb/...

Typical Windows PowerShell usage:
  cd $HOME\Downloads\mteb
  python .\script\bench_mteb.py --download-only
  python .\script\bench_mteb.py --benchmark "MTEB(eng, v2)" --languages eng --save-predictions

Notes:
  * Default inference uses native Hugging Face Transformers, not FlagEmbedding.
    This avoids tokenizer.prepare_for_model compatibility failures in newer Transformers releases.
  * MedSwin and BGE v2 Gemma are scored as LLM/CausalLM rerankers.
  * BGE v2 M3 is scored as a SequenceClassification reranker.
  * MTEB computes the official reranking metrics; this script exports every
    numeric metric returned by MTEB into JSONL/CSV, plus cross-model comparison CSVs.

Key metrics:
Main score | NDCG@10 | MAP@10 | MRR@10 | Recall@10

"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import logging
import math
import os
import re
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("medswin_mteb_rerank")


# -----------------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    repo_id: str
    local_dir_name: str
    backend: str  # "llm" for Gemma-style causal LM reranker, "sequence" for sequence-classification reranker
    default_batch_size: int
    default_max_length: int | None = None
    trust_remote_code: bool = True


MODEL_SPECS: dict[str, ModelSpec] = {
    "medswin-reranker-bge-gemma": ModelSpec(
        key="medswin-reranker-bge-gemma",
        display_name="MedSwin-Reranker-bge-gemma",
        repo_id="MedSwin/MedSwin-Reranker-bge-gemma",
        local_dir_name="medswin-reranker-bge-gemma",
        backend="llm",
        default_batch_size=4,
        default_max_length=None,
    ),
    "bge-reranker-v2-gemma": ModelSpec(
        key="bge-reranker-v2-gemma",
        display_name="BGE-Reranker-v2-Gemma",
        repo_id="BAAI/bge-reranker-v2-gemma",
        local_dir_name="bge-reranker-v2-gemma",
        backend="llm",
        default_batch_size=4,
        default_max_length=None,
    ),
    "bge-reranker-v2-m3": ModelSpec(
        key="bge-reranker-v2-m3",
        display_name="BGE-Reranker-v2-M3",
        repo_id="BAAI/bge-reranker-v2-m3",
        local_dir_name="bge-reranker-v2-m3",
        backend="sequence",
        default_batch_size=32,
        default_max_length=1024,
    ),
}

MODEL_ALIASES = {
    "all": "all",
    "medswin": "medswin-reranker-bge-gemma",
    "medswin-gemma": "medswin-reranker-bge-gemma",
    "medswin-reranker": "medswin-reranker-bge-gemma",
    "bge-gemma": "bge-reranker-v2-gemma",
    "gemma": "bge-reranker-v2-gemma",
    "bge-v2-gemma": "bge-reranker-v2-gemma",
    "bge-m3": "bge-reranker-v2-m3",
    "m3": "bge-reranker-v2-m3",
    "bge-v2-m3": "bge-reranker-v2-m3",
}


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def project_root_from_script() -> Path:
    # Expected: Downloads/mteb/script/bench_mteb.py => root is parent of script dir.
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        return Path.cwd().resolve()


def now_run_name() -> str:
    return dt.datetime.now().strftime("mteb_rerank_%Y%m%d_%H%M%S")


def safe_name(value: str, max_len: int = 120) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("._-")
    return (value or "item")[:max_len]


def comma_or_space_list(values: Sequence[str] | None) -> list[str] | None:
    if not values:
        return None
    out: list[str] = []
    for item in values:
        for part in str(item).replace(",", " ").split():
            part = part.strip()
            if part:
                out.append(part)
    return out or None


def parse_models(values: Sequence[str] | None) -> list[ModelSpec]:
    raw = comma_or_space_list(values) or ["all"]
    keys: list[str] = []
    for item in raw:
        canonical = MODEL_ALIASES.get(item.lower(), item)
        if canonical == "all":
            keys.extend(MODEL_SPECS.keys())
        elif canonical in MODEL_SPECS:
            keys.append(canonical)
        else:
            valid = ", ".join(sorted(set(MODEL_SPECS) | set(MODEL_ALIASES)))
            raise SystemExit(f"Unknown model selector '{item}'. Valid choices include: {valid}")

    # Stable de-duplication preserving registry order.
    selected = []
    seen = set()
    for key in MODEL_SPECS:
        if key in keys and key not in seen:
            selected.append(MODEL_SPECS[key])
            seen.add(key)
    return selected


def model_local_dir(root: Path, spec: ModelSpec) -> Path:
    return root / "model" / spec.local_dir_name


def has_model_files(path: Path) -> bool:
    if not path.exists():
        return False
    if not (path / "config.json").exists():
        return False
    weight_patterns = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf"]
    return any(path.glob(pattern) for pattern in weight_patterns)


def set_repro_and_perf_env(root: Path, hf_transfer: bool) -> None:
    # Keep caches inside Downloads/mteb/cache by default.
    cache_root = root / "cache"
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "huggingface" / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "huggingface" / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "huggingface" / "transformers"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if hf_transfer:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Download handling
# -----------------------------------------------------------------------------


def download_model(
    spec: ModelSpec,
    root: Path,
    *,
    revision: str | None,
    token: str | bool | None,
    force_download: bool,
    local_files_only: bool,
) -> Path:
    local_dir = model_local_dir(root, spec)
    if has_model_files(local_dir) and not force_download:
        LOGGER.info("Model already present: %s -> %s", spec.repo_id, local_dir)
        return local_dir

    if local_files_only:
        raise FileNotFoundError(
            f"{spec.display_name} is not present at {local_dir}, and --local-files-only was used."
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("Install huggingface_hub first: python -m pip install -U huggingface_hub") from exc

    LOGGER.info("Downloading %s to %s", spec.repo_id, local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Avoid downloading optional duplicate framework files if present. Keep all safetensors shards.
    ignore_patterns = [
        "*.msgpack",
        "*.h5",
        "*.ot",
        "*.onnx",
        "*.tflite",
        "*.tar.gz",
    ]

    snapshot_download(
        repo_id=spec.repo_id,
        revision=revision,
        local_dir=str(local_dir),
        token=token,
        force_download=force_download,
        local_files_only=False,
        ignore_patterns=ignore_patterns,
    )

    if not has_model_files(local_dir):
        raise RuntimeError(f"Download completed, but expected config/weight files were not found in {local_dir}")
    return local_dir


def download_selected_models(args: argparse.Namespace, specs: list[ModelSpec], root: Path) -> dict[str, Path]:
    token: str | bool | None
    if args.hf_token is None:
        token = None
    elif args.hf_token.lower() in {"1", "true", "yes"}:
        token = True
    else:
        token = args.hf_token

    out: dict[str, Path] = {}
    for spec in specs:
        out[spec.key] = download_model(
            spec,
            root,
            revision=args.revision,
            token=token,
            force_download=args.force_download,
            local_files_only=args.local_files_only,
        )
    return out


# -----------------------------------------------------------------------------
# MTEB input conversion helpers
# -----------------------------------------------------------------------------


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return record_to_text(value)
    if isinstance(value, (list, tuple)):
        return "\n".join(to_text(x) for x in value if x is not None)
    return str(value)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
            return converted if isinstance(converted, list) else [converted]
        except Exception:
            pass
    return [value]


def record_to_text(record: Mapping[str, Any]) -> str:
    if "text" in record:
        text = to_text(record.get("text"))
        title = to_text(record.get("title")) if record.get("title") is not None else ""
        if title and title not in text[: max(200, len(title))]:
            return f"{title}\n{text}".strip()
        return text

    title = to_text(record.get("title")) if record.get("title") is not None else ""
    body = to_text(record.get("body")) if record.get("body") is not None else ""
    if title or body:
        return f"{title}\n{body}".strip()

    ignore = {"id", "qid", "query-id", "corpus-id", "score", "label", "labels"}
    pieces = [to_text(v) for k, v in record.items() if k not in ignore and v is not None]
    return "\n".join(pieces) if pieces else ""


def batch_to_texts(batch: Any) -> list[str]:
    """Convert MTEB BatchedInput/DataLoader batches into list[str]."""
    if isinstance(batch, Mapping):
        if "text" in batch:
            texts = [to_text(x) for x in as_list(batch.get("text"))]
            titles = as_list(batch.get("title")) if "title" in batch else []
            if titles and len(titles) == len(texts):
                out = []
                for title, text in zip(titles, texts, strict=False):
                    title_s = to_text(title)
                    if title_s and title_s not in text[: max(200, len(title_s))]:
                        out.append(f"{title_s}\n{text}".strip())
                    else:
                        out.append(text)
                return out
            return texts

        if "body" in batch or "title" in batch:
            titles = as_list(batch.get("title"))
            bodies = as_list(batch.get("body"))
            n = max(len(titles), len(bodies))
            if len(titles) == 1 and n > 1:
                titles *= n
            if len(bodies) == 1 and n > 1:
                bodies *= n
            return [f"{to_text(t)}\n{to_text(b)}".strip() for t, b in zip(titles, bodies, strict=False)]

        return [record_to_text(batch)]

    if isinstance(batch, str):
        return [batch]

    if isinstance(batch, (list, tuple)):
        if not batch:
            return []
        if all(isinstance(x, Mapping) for x in batch):
            return [record_to_text(x) for x in batch]
        if all(isinstance(x, str) for x in batch):
            return list(batch)
        return [to_text(x) for x in batch]

    return [to_text(batch)]


def numeric_array(scores: Any) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr



# -----------------------------------------------------------------------------
# MTEB CrossEncoderProtocol wrappers
# -----------------------------------------------------------------------------


def chunked(seq: list[Any], size: int) -> Iterable[list[Any]]:
    """Yield non-empty chunks from a list."""
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def sigmoid_array(values: np.ndarray) -> np.ndarray:
    # Stable enough for reranker logits; values are usually small.
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def torch_device_or_raise(device: str | None) -> str:
    """Resolve a device string and fail early for accidental CPU-only CUDA runs."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required. Install a CUDA build of torch for RTX 4090 runs.") from exc

    if not device or device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "You requested a CUDA device, but torch.cuda.is_available() is False. "
            "Your environment likely has a CPU-only torch wheel. Reinstall PyTorch with a CUDA index URL, "
            "then rerun the benchmark."
        )
    return device


class MTEBModelMetaMixin:
    def _build_mteb_model_meta(
        self,
        *,
        display_name: str,
        model_name_or_path: str,
        revision: str | None,
        max_length: int | None,
        framework: list[str],
        model_type: list[str],
    ) -> Any:
        try:
            from mteb.models import ModelMeta
        except Exception:
            try:
                from mteb import ModelMeta  # type: ignore
            except Exception:
                return None

        reference = None
        if "/" in model_name_or_path and not Path(model_name_or_path).exists():
            reference = f"https://huggingface.co/{model_name_or_path}"

        try:
            return ModelMeta.create_empty(
                overwrites={
                    "name": display_name,
                    "revision": revision,
                    "framework": framework,
                    "model_type": model_type,
                    "reference": reference,
                    "max_tokens": max_length,
                    "open_weights": True,
                }
            )
        except Exception:
            return None


class TransformersRerankerForMTEB(MTEBModelMetaMixin):
    """
    Native Transformers implementation of MTEB's CrossEncoderProtocol.

    This is now the default path because recent Transformers tokenizer changes
    can break FlagEmbedding's older tokenizer.prepare_for_model calls. The
    implementation below follows the BGE model-card inference recipes, but
    deliberately avoids prepare_for_model.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        architecture: str,  # "sequence" or "llm"
        display_name: str,
        revision: str | None = None,
        device: str | None = None,
        use_fp16: bool = True,
        use_bf16: bool = False,
        normalize: bool = False,
        batch_size: int = 4,
        max_length: int | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
        trust_remote_code: bool = True,
        cache_dir: str | None = None,
        llm_prompt: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Transformers and PyTorch are required. Install with: "
                "python -m pip install -U transformers accelerate torch"
            ) from exc

        self.torch = torch
        self.model_name_or_path = model_name_or_path
        self.architecture = architecture
        self.display_name = display_name
        self.revision = revision
        self.batch_size = int(batch_size)
        self.max_length = int(max_length or (1024 if architecture == "llm" else 512))
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.device = torch_device_or_raise(device)
        self._meta = None
        self.llm_prompt = llm_prompt or (
            "Given a query A and a passage B, determine whether the passage contains an answer "
            "to the query by providing a prediction of either 'Yes' or 'No'."
        )

        if use_bf16:
            dtype = torch.bfloat16
        elif use_fp16 and self.device.startswith("cuda"):
            dtype = torch.float16
        else:
            dtype = torch.float32
        self.dtype = dtype

        tokenizer_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "cache_dir": cache_dir,
        }
        if revision:
            tokenizer_kwargs["revision"] = revision

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "cache_dir": cache_dir,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if revision:
            model_kwargs["revision"] = revision

        if architecture == "sequence":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **model_kwargs)
        elif architecture == "llm":
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
            # Decoder-only reranker scoring reads logits at the final non-padding position. Left padding
            # keeps `logits[:, -1, yes_token]` aligned for every row in a padded batch.
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                elif self.tokenizer.bos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.bos_token
            if getattr(self.model.config, "pad_token_id", None) is None and self.tokenizer.pad_token_id is not None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

            yes_ids = self.tokenizer("Yes", add_special_tokens=False)["input_ids"]
            if not yes_ids:
                raise RuntimeError("Could not resolve token id for 'Yes' in the LLM reranker tokenizer.")
            self.yes_token_id = int(yes_ids[0])
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.model.to(self.device)
        self.model.eval()

    @property
    def mteb_model_meta(self) -> Any:
        if self._meta is None:
            self._meta = self._build_mteb_model_meta(
                display_name=self.display_name,
                model_name_or_path=self.model_name_or_path,
                revision=self.revision,
                max_length=self.max_length,
                framework=["PyTorch", "Transformers"],
                model_type=["cross-encoder", self.architecture],
            )
        return self._meta

    def _sequence_scores(self, pairs: list[list[str]], batch_size: int) -> np.ndarray:
        if not pairs:
            return np.asarray([], dtype=np.float32)

        scores_out: list[np.ndarray] = []
        for batch_pairs in chunked(pairs, batch_size):
            with self.torch.no_grad():
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                ).to(self.device)
                logits = self.model(**inputs, return_dict=True).logits.float()
                if logits.ndim == 2 and logits.shape[-1] > 1:
                    batch_scores = logits[:, -1]
                else:
                    batch_scores = logits.view(-1)
                arr = batch_scores.detach().cpu().numpy().astype(np.float32)
                if self.normalize:
                    arr = sigmoid_array(arr)
                scores_out.append(arr)

        return np.concatenate(scores_out).astype(np.float32)

    def _llm_inputs_for_pairs(self, pairs: list[list[str]]) -> Any:
        """Build BGE Gemma reranker inputs without tokenizer.prepare_for_model."""
        sep = "\n"
        tokenizer = self.tokenizer
        prompt_ids = tokenizer(self.llm_prompt, return_tensors=None, add_special_tokens=False)["input_ids"]
        sep_ids = tokenizer(sep, return_tensors=None, add_special_tokens=False)["input_ids"]

        encoded_items: list[dict[str, list[int]]] = []
        query_max_length = self.max_length * 3 // 4

        bos_id = tokenizer.bos_token_id
        for query, passage in pairs:
            query_ids = tokenizer(
                f"A: {query}",
                return_tensors=None,
                add_special_tokens=False,
                max_length=query_max_length,
                truncation=True,
            )["input_ids"]
            passage_ids = tokenizer(
                f"B: {passage}",
                return_tensors=None,
                add_special_tokens=False,
                max_length=self.max_length,
                truncation=True,
            )["input_ids"]

            first_ids: list[int] = ([] if bos_id is None else [int(bos_id)]) + list(query_ids)
            first_ids = first_ids + list(sep_ids)
            second_ids = list(sep_ids) + list(passage_ids)

            # Emulate prepare_for_model(..., truncation="only_second", max_length=self.max_length).
            second_budget = max(0, self.max_length - len(first_ids))
            if len(second_ids) > second_budget:
                second_ids = second_ids[:second_budget]

            input_ids = first_ids + second_ids + list(sep_ids) + list(prompt_ids)
            encoded_items.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})

        padded = tokenizer.pad(
            encoded_items,
            padding=True,
            max_length=self.max_length + len(sep_ids) + len(prompt_ids),
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        return padded.to(self.device)

    def _llm_scores(self, pairs: list[list[str]], batch_size: int) -> np.ndarray:
        if not pairs:
            return np.asarray([], dtype=np.float32)

        scores_out: list[np.ndarray] = []
        for batch_pairs in chunked(pairs, batch_size):
            with self.torch.no_grad():
                inputs = self._llm_inputs_for_pairs(batch_pairs)
                logits = self.model(**inputs, return_dict=True).logits
                batch_scores = logits[:, -1, self.yes_token_id].view(-1).float()
                arr = batch_scores.detach().cpu().numpy().astype(np.float32)
                if self.normalize:
                    arr = sigmoid_array(arr)
                scores_out.append(arr)

        return np.concatenate(scores_out).astype(np.float32)

    def _score_pairs(self, pairs: list[list[str]], batch_size: int) -> np.ndarray:
        if self.architecture == "sequence":
            return self._sequence_scores(pairs, batch_size)
        if self.architecture == "llm":
            return self._llm_scores(pairs, batch_size)
        raise ValueError(f"Unsupported architecture: {self.architecture}")

    def predict(
        self,
        inputs1: Iterable[Any],
        inputs2: Iterable[Any],
        *,
        task_metadata: Any = None,
        hf_split: str = "test",
        hf_subset: str = "default",
        prompt_type: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        batch_size = int(kwargs.get("batch_size") or self.batch_size)
        all_scores: list[np.ndarray] = []
        seen_pairs = 0

        for batch1, batch2 in zip(inputs1, inputs2, strict=False):
            queries = batch_to_texts(batch1)
            docs = batch_to_texts(batch2)

            if len(queries) != len(docs):
                if len(queries) == 1:
                    queries = queries * len(docs)
                elif len(docs) == 1:
                    docs = docs * len(queries)
                else:
                    task_name_for_error = getattr(task_metadata, "name", task_metadata)
                    raise ValueError(
                        f"Mismatched query/doc batch sizes: {len(queries)} queries vs {len(docs)} docs "
                        f"for task={task_name_for_error}, split={hf_split}, subset={hf_subset}"
                    )

            pairs = [
                [f"{self.query_prefix}{q}", f"{self.document_prefix}{d}"]
                for q, d in zip(queries, docs, strict=True)
            ]
            scores = self._score_pairs(pairs, batch_size=batch_size)
            seen_pairs += len(pairs)
            all_scores.append(scores)

        if not all_scores:
            LOGGER.warning("No pairs were scored for task=%s split=%s subset=%s", task_metadata, hf_split, hf_subset)
            return np.asarray([], dtype=np.float32)

        output = np.concatenate(all_scores).astype(np.float32)
        LOGGER.debug("Scored %s query-document pairs", seen_pairs)
        return output


class FlagRerankerForMTEB(MTEBModelMetaMixin):
    """Optional legacy adapter for FlagEmbedding. Prefer --implementation transformers."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        architecture: str,
        display_name: str,
        revision: str | None = None,
        device: str | None = None,
        devices: list[str] | None = None,
        use_fp16: bool = True,
        use_bf16: bool = False,
        normalize: bool = False,
        batch_size: int = 4,
        max_length: int | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
        trust_remote_code: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        try:
            import FlagEmbedding as flag_embedding
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is required. Install with: python -m pip install -U FlagEmbedding"
            ) from exc

        cls_name = "FlagLLMReranker" if architecture == "llm" else "FlagReranker"
        RerankerClass = getattr(flag_embedding, cls_name)
        self.model_name_or_path = model_name_or_path
        self.architecture = architecture
        self.display_name = display_name
        self.revision = revision
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._meta = None

        init_kwargs: dict[str, Any] = {}
        if use_bf16:
            init_kwargs["use_bf16"] = True
        else:
            init_kwargs["use_fp16"] = use_fp16
        if devices:
            init_kwargs["devices"] = devices
        elif device:
            init_kwargs["devices"] = [device]
        if trust_remote_code is not None:
            init_kwargs["trust_remote_code"] = trust_remote_code
        if cache_dir:
            init_kwargs["cache_dir"] = cache_dir
        if revision:
            init_kwargs["revision"] = revision

        try:
            self.reranker = RerankerClass(model_name_or_path, **init_kwargs)
        except TypeError as first_exc:
            LOGGER.debug("Full init kwargs failed for %s: %s", cls_name, first_exc)
            minimal_kwargs: dict[str, Any] = {}
            if use_bf16:
                minimal_kwargs["use_bf16"] = True
            else:
                minimal_kwargs["use_fp16"] = use_fp16
            try:
                self.reranker = RerankerClass(model_name_or_path, **minimal_kwargs)
            except TypeError:
                self.reranker = RerankerClass(model_name_or_path)

    @property
    def mteb_model_meta(self) -> Any:
        if self._meta is None:
            self._meta = self._build_mteb_model_meta(
                display_name=self.display_name,
                model_name_or_path=self.model_name_or_path,
                revision=self.revision,
                max_length=self.max_length,
                framework=["PyTorch", "Transformers", "FlagEmbedding"],
                model_type=["cross-encoder", self.architecture],
            )
        return self._meta

    def _score_pairs(self, pairs: list[list[str]], batch_size: int) -> np.ndarray:
        if not pairs:
            return np.asarray([], dtype=np.float32)

        kwargs: dict[str, Any] = {"batch_size": batch_size}
        if self.max_length is not None:
            kwargs["max_length"] = self.max_length
        if self.normalize is not None:
            kwargs["normalize"] = self.normalize

        try:
            scores = self.reranker.compute_score(pairs, **kwargs)
        except TypeError:
            try:
                scores = self.reranker.compute_score(pairs, batch_size=batch_size)
            except TypeError:
                scores = self.reranker.compute_score(pairs)

        arr = numeric_array(scores)
        if len(arr) != len(pairs):
            raise RuntimeError(f"Reranker returned {len(arr)} scores for {len(pairs)} pairs")
        return arr

    def predict(
        self,
        inputs1: Iterable[Any],
        inputs2: Iterable[Any],
        *,
        task_metadata: Any = None,
        hf_split: str = "test",
        hf_subset: str = "default",
        prompt_type: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        batch_size = int(kwargs.get("batch_size") or self.batch_size)
        all_scores: list[np.ndarray] = []
        seen_pairs = 0

        for batch1, batch2 in zip(inputs1, inputs2, strict=False):
            queries = batch_to_texts(batch1)
            docs = batch_to_texts(batch2)

            if len(queries) != len(docs):
                if len(queries) == 1:
                    queries = queries * len(docs)
                elif len(docs) == 1:
                    docs = docs * len(queries)
                else:
                    task_name_for_error = getattr(task_metadata, "name", task_metadata)
                    raise ValueError(
                        f"Mismatched query/doc batch sizes: {len(queries)} queries vs {len(docs)} docs "
                        f"for task={task_name_for_error}, split={hf_split}, subset={hf_subset}"
                    )

            pairs = [
                [f"{self.query_prefix}{q}", f"{self.document_prefix}{d}"]
                for q, d in zip(queries, docs, strict=True)
            ]
            scores = self._score_pairs(pairs, batch_size=batch_size)
            seen_pairs += len(pairs)
            all_scores.append(scores)

        if not all_scores:
            LOGGER.warning("No pairs were scored for task=%s split=%s subset=%s", task_metadata, hf_split, hf_subset)
            return np.asarray([], dtype=np.float32)

        output = np.concatenate(all_scores).astype(np.float32)
        LOGGER.debug("Scored %s query-document pairs", seen_pairs)
        return output


def load_reranker_for_mteb(
    spec: ModelSpec,
    local_dir: Path,
    args: argparse.Namespace,
    root: Path,
) -> Any:
    batch_size = args.batch_size or spec.default_batch_size
    max_length = args.max_length if args.max_length is not None else spec.default_max_length
    devices = comma_or_space_list(args.devices)

    if args.implementation == "flagembedding":
        try:
            import transformers
            version = getattr(transformers, "__version__", "unknown")
            if str(version).split(".", 1)[0].isdigit() and int(str(version).split(".", 1)[0]) >= 5:
                LOGGER.warning(
                    "You selected --implementation flagembedding with transformers %s. "
                    "If you see '*Tokenizer has no attribute prepare_for_model', use the default "
                    "--implementation transformers or pin transformers<5.",
                    version,
                )
        except Exception:
            pass
        return FlagRerankerForMTEB(
            str(local_dir),
            architecture=spec.backend,
            display_name=spec.display_name,
            revision=args.revision,
            device=args.device,
            devices=devices,
            use_fp16=args.use_fp16,
            use_bf16=args.use_bf16,
            normalize=args.normalize_scores,
            batch_size=batch_size,
            max_length=max_length,
            query_prefix=args.query_prefix,
            document_prefix=args.document_prefix,
            trust_remote_code=spec.trust_remote_code,
            cache_dir=str(root / "cache" / "huggingface"),
        )

    if devices:
        LOGGER.warning("--devices is only used by --implementation flagembedding; native Transformers uses --device.")

    return TransformersRerankerForMTEB(
        str(local_dir),
        architecture=spec.backend,
        display_name=spec.display_name,
        revision=args.revision,
        device=args.device,
        use_fp16=args.use_fp16,
        use_bf16=args.use_bf16,
        normalize=args.normalize_scores,
        batch_size=batch_size,
        max_length=max_length,
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
        trust_remote_code=spec.trust_remote_code,
        cache_dir=str(root / "cache" / "huggingface"),
        llm_prompt=args.llm_prompt,
    )


# -----------------------------------------------------------------------------
# MTEB task selection
# -----------------------------------------------------------------------------


def task_metadata(task: Any) -> Any:
    return getattr(task, "metadata", None)


def task_name(task: Any) -> str:
    meta = task_metadata(task)
    return getattr(meta, "name", None) or getattr(task, "name", None) or task.__class__.__name__


def task_type(task: Any) -> str | None:
    meta = task_metadata(task)
    return getattr(meta, "type", None) or getattr(task, "type", None)


def task_domains(task: Any) -> list[str]:
    meta = task_metadata(task)
    domains = getattr(meta, "domains", None) if meta is not None else None
    return list(domains) if domains else []


def task_languages(task: Any) -> list[str]:
    meta = task_metadata(task)
    langs = getattr(meta, "languages", None) if meta is not None else None
    return list(langs) if langs else []


def filter_task_list(
    tasks: Iterable[Any],
    *,
    domains: list[str] | None = None,
    languages: list[str] | None = None,
    task_types: list[str] | None = None,
) -> list[Any]:
    domain_set = {x.lower() for x in (domains or [])}
    lang_set = {x.lower() for x in (languages or [])}
    type_set = set(task_types or [])
    out = []
    for task in tasks:
        if type_set and task_type(task) not in type_set:
            continue
        if domain_set and not domain_set.intersection({x.lower() for x in task_domains(task)}):
            continue
        if lang_set and not lang_set.intersection({x.lower() for x in task_languages(task)}):
            continue
        out.append(task)
    return out


def build_tasks(args: argparse.Namespace) -> list[Any]:
    import mteb

    languages = comma_or_space_list(args.languages)
    domains = comma_or_space_list(args.domains)
    if args.medical_only:
        domains = sorted(set((domains or []) + ["Medical"]))

    eval_splits = comma_or_space_list(args.eval_splits)

    if args.tasks:
        names = comma_or_space_list(args.tasks) or []
        tasks: list[Any] = []
        for name in names:
            try:
                if eval_splits:
                    tasks.append(mteb.get_task(name, eval_splits=eval_splits))
                else:
                    tasks.append(mteb.get_task(name))
            except Exception:
                tasks.extend(mteb.get_tasks(tasks=[name]))
        return filter_task_list(tasks, domains=domains, languages=languages, task_types=["Reranking"])

    if args.benchmark:
        benchmark = mteb.get_benchmark(args.benchmark)
        bench_tasks = getattr(benchmark, "tasks", benchmark)
        tasks = filter_task_list(list(bench_tasks), domains=domains, languages=languages, task_types=["Reranking"])
    else:
        kwargs: dict[str, Any] = {"task_types": ["Reranking"]}
        if languages:
            kwargs["languages"] = languages
        if domains:
            kwargs["domains"] = domains
        if args.text_only:
            kwargs["modalities"] = ["text"]
        try:
            tasks = list(mteb.get_tasks(**kwargs))
        except TypeError:
            tasks = list(mteb.get_tasks(task_types=["Reranking"]))
            tasks = filter_task_list(tasks, domains=domains, languages=languages, task_types=["Reranking"])

    if eval_splits:
        rebuilt: list[Any] = []
        for task in tasks:
            try:
                rebuilt.append(mteb.get_task(task_name(task), eval_splits=eval_splits))
            except Exception:
                rebuilt.append(task)
        tasks = rebuilt

    tasks = sorted(tasks, key=task_name)
    if args.limit_tasks is not None:
        tasks = tasks[: args.limit_tasks]
    return tasks


def task_manifest(tasks: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": task_name(task),
            "type": task_type(task),
            "domains": task_domains(task),
            "languages": task_languages(task),
        }
        for task in tasks
    ]


# -----------------------------------------------------------------------------
# Result serialization and flattening
# -----------------------------------------------------------------------------


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if hasattr(obj, "model_dump"):
        try:
            return to_jsonable(obj.model_dump(mode="json"))
        except TypeError:
            return to_jsonable(obj.model_dump())
    if hasattr(obj, "dict"):
        try:
            return to_jsonable(obj.dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__") and obj.__class__.__module__.startswith("mteb"):
        return to_jsonable(vars(obj))
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    return str(obj)


def is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, np.number)):
        try:
            return not (math.isnan(float(value)) or math.isinf(float(value)))
        except Exception:
            return True
    return False


def walk_numeric(obj: Any, prefix: str = "") -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    if is_number(obj):
        rows.append((prefix.strip("."), float(obj)))
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(walk_numeric(value, next_prefix))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            rows.extend(walk_numeric(value, next_prefix))
    return rows


def looks_like_task_result(obj: Mapping[str, Any]) -> bool:
    keys = set(obj.keys())
    return bool(keys.intersection({"task_name", "scores", "results", "metadata", "task"}))


def infer_task_name_from_result(obj: Mapping[str, Any], fallback: str = "unknown_task") -> str:
    for key in ("task_name", "name"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    meta = obj.get("metadata")
    if isinstance(meta, Mapping):
        value = meta.get("name")
        if isinstance(value, str) and value:
            return value
    task = obj.get("task")
    if isinstance(task, Mapping):
        value = task.get("name")
        if isinstance(value, str) and value:
            return value
    return fallback


def flatten_one_result(result_data: Any, *, model_key: str, model_display_name: str, fallback_task: str) -> list[dict[str, Any]]:
    data = to_jsonable(result_data)
    rows: list[dict[str, Any]] = []

    if isinstance(data, list):
        for item in data:
            rows.extend(
                flatten_one_result(
                    item,
                    model_key=model_key,
                    model_display_name=model_display_name,
                    fallback_task=fallback_task,
                )
            )
        return rows

    if not isinstance(data, Mapping):
        return rows

    # Common MTEB aggregate shape.
    if isinstance(data.get("task_results"), list):
        for item in data["task_results"]:
            rows.extend(
                flatten_one_result(
                    item,
                    model_key=model_key,
                    model_display_name=model_display_name,
                    fallback_task=fallback_task,
                )
            )
        return rows

    task = infer_task_name_from_result(data, fallback=fallback_task)
    scores = data.get("scores")
    if scores is None:
        scores = data.get("results")
    if scores is None:
        # Some versions put numbers directly on the result object.
        scores = {
            k: v
            for k, v in data.items()
            if k not in {"model", "task", "metadata", "task_name", "name", "hf_subset", "scores"}
        }

    for metric_path, value in walk_numeric(scores):
        if not metric_path:
            continue
        rows.append(
            {
                "model_key": model_key,
                "model": model_display_name,
                "task": task,
                "metric_path": metric_path,
                "metric_leaf": metric_path.split(".")[-1],
                "value": value,
            }
        )
    return rows


def write_metric_outputs(model_out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(model_out_dir / "metrics_long.jsonl", rows)
    write_csv(model_out_dir / "metrics_long.csv", rows)

    try:
        import pandas as pd
    except ImportError:
        LOGGER.warning("pandas not installed; skipped wide/summary CSV exports")
        return

    df = pd.DataFrame(rows)
    if df.empty:
        (model_out_dir / "metrics_wide.csv").write_text("", encoding="utf-8")
        (model_out_dir / "metric_summary.csv").write_text("", encoding="utf-8")
        return

    wide = df.pivot_table(index=["task"], columns=["metric_path"], values="value", aggfunc="first")
    wide.reset_index().to_csv(model_out_dir / "metrics_wide.csv", index=False)

    summary = (
        df.groupby(["metric_path", "metric_leaf"], as_index=False)["value"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .sort_values(["metric_path"])
    )
    summary.to_csv(model_out_dir / "metric_summary.csv", index=False)


def write_cross_model_outputs(run_dir: Path, all_rows: list[dict[str, Any]], failed_rows: list[dict[str, Any]]) -> None:
    write_jsonl(run_dir / "all_models_metrics_long.jsonl", all_rows)
    write_csv(run_dir / "all_models_metrics_long.csv", all_rows)
    write_json(run_dir / "failed_tasks.json", failed_rows)

    try:
        import pandas as pd
    except ImportError:
        LOGGER.warning("pandas not installed; skipped cross-model wide/summary CSV exports")
        return

    df = pd.DataFrame(all_rows)
    if df.empty:
        return

    comparison = df.pivot_table(
        index=["task", "metric_path", "metric_leaf"],
        columns="model",
        values="value",
        aggfunc="first",
    ).reset_index()
    comparison.to_csv(run_dir / "comparison_by_task_metric.csv", index=False)

    by_model_metric = (
        df.groupby(["model", "metric_path", "metric_leaf"], as_index=False)["value"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .sort_values(["metric_path", "model"])
    )
    by_model_metric.to_csv(run_dir / "summary_by_model_metric.csv", index=False)

    # A compact task count and metric count summary.
    compact = (
        df.groupby("model", as_index=False)
        .agg(tasks=("task", "nunique"), metric_values=("value", "count"))
        .sort_values("model")
    )
    compact.to_csv(run_dir / "model_run_summary.csv", index=False)


# -----------------------------------------------------------------------------
# MTEB evaluation
# -----------------------------------------------------------------------------


def make_result_cache(cache_path: Path) -> Any:
    import mteb

    try:
        return mteb.ResultCache(cache_path=cache_path)
    except TypeError:
        return mteb.ResultCache(cache_path)


def evaluate_task_compat(model: Any, task: Any, args: argparse.Namespace, cache: Any, prediction_folder: Path | None) -> Any:
    import mteb

    encode_kwargs = {"batch_size": int(args.batch_size or 1)}
    kwargs: dict[str, Any] = {
        "tasks": [task],
        "cache": cache,
        "overwrite_strategy": args.overwrite_strategy,
        "prediction_folder": prediction_folder,
        "show_progress_bar": True,
        "public_only": args.public_only,
        "num_proc": args.num_proc,
        "co2_tracker": args.co2_tracker,
        "encode_kwargs": encode_kwargs,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        return mteb.evaluate(model, **kwargs)
    except TypeError as exc:
        # Compatibility fallback for MTEB versions with slightly different keyword support.
        message = str(exc)
        removable = [
            "prediction_folder",
            "public_only",
            "num_proc",
            "co2_tracker",
            "overwrite_strategy",
            "encode_kwargs",
            "cache",
        ]
        if "unexpected keyword" not in message and "got an unexpected" not in message:
            raise
        reduced = dict(kwargs)
        changed = False
        for key in removable:
            if key in reduced and key in message:
                reduced.pop(key, None)
                changed = True
        if not changed:
            # Conservative broad fallback: remove optional keys but keep tasks.
            for key in removable:
                reduced.pop(key, None)
        LOGGER.warning("Retrying mteb.evaluate with reduced kwargs after TypeError: %s", message)
        return mteb.evaluate(model, **reduced)


def unload_model(model: Any) -> None:
    try:
        del model
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def run_one_model(
    spec: ModelSpec,
    local_dir: Path,
    tasks: list[Any],
    args: argparse.Namespace,
    root: Path,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_out_dir = run_dir / "models" / spec.key
    model_out_dir.mkdir(parents=True, exist_ok=True)
    write_json(model_out_dir / "model_spec.json", asdict(spec) | {"local_dir": str(local_dir)})

    LOGGER.info("Loading %s from %s", spec.display_name, local_dir)
    model = load_reranker_for_mteb(spec, local_dir, args, root)

    cache = None if args.no_cache else make_result_cache(root / "cache" / "mteb" / spec.key)
    prediction_folder = model_out_dir / "predictions" if args.save_predictions else None
    if prediction_folder:
        prediction_folder.mkdir(parents=True, exist_ok=True)

    task_results_dir = model_out_dir / "task_results"
    task_results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    model_started = time.time()

    for index, task in enumerate(tasks, start=1):
        name = task_name(task)
        LOGGER.info("[%s/%s] %s :: %s", index, len(tasks), spec.display_name, name)
        task_started = time.time()
        try:
            result = evaluate_task_compat(model, task, args, cache, prediction_folder)
            elapsed = time.time() - task_started
            result_json = {
                "task": name,
                "elapsed_seconds": elapsed,
                "result": to_jsonable(result),
            }
            write_json(task_results_dir / f"{safe_name(name)}.json", result_json)
            rows = flatten_one_result(
                result,
                model_key=spec.key,
                model_display_name=spec.display_name,
                fallback_task=name,
            )
            for row in rows:
                row["task_elapsed_seconds"] = elapsed
            all_rows.extend(rows)
        except Exception as exc:
            elapsed = time.time() - task_started
            failure = {
                "model_key": spec.key,
                "model": spec.display_name,
                "task": name,
                "elapsed_seconds": elapsed,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
            failed_rows.append(failure)
            write_json(task_results_dir / f"{safe_name(name)}__FAILED.json", failure)
            LOGGER.exception("Task failed for %s / %s", spec.display_name, name)
            if not args.continue_on_error:
                raise

    elapsed_total = time.time() - model_started
    write_json(
        model_out_dir / "run_metadata.json",
        {
            "model": asdict(spec),
            "local_dir": str(local_dir),
            "elapsed_seconds": elapsed_total,
            "num_tasks": len(tasks),
            "num_failed_tasks": len(failed_rows),
            "args": sanitize_args(vars(args)),
        },
    )
    write_metric_outputs(model_out_dir, all_rows)
    write_json(model_out_dir / "failed_tasks.json", failed_rows)

    unload_model(model)
    return all_rows, failed_rows


def sanitize_args(args_dict: dict[str, Any]) -> dict[str, Any]:
    clean = dict(args_dict)
    if clean.get("hf_token"):
        clean["hf_token"] = "***"
    return clean


def print_cuda_summary() -> None:
    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA runtime: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"GPU memory: {props.total_memory / (1024**3):.1f} GiB")
    except Exception as exc:
        print(f"Could not inspect CUDA/PyTorch: {exc}")

    try:
        import transformers

        print(f"Transformers: {transformers.__version__}")
    except Exception:
        print("Transformers: not importable")

    try:
        import mteb

        print(f"MTEB: {getattr(mteb, '__version__', 'unknown')}")
    except Exception:
        print("MTEB: not importable")

    try:
        import FlagEmbedding

        print(f"FlagEmbedding: {getattr(FlagEmbedding, '__version__', 'unknown')}")
    except Exception:
        print("FlagEmbedding: not importable")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and benchmark MedSwin/BGE rerankers on MTEB reranking tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", default=None, help="Project root. Defaults to parent of this script directory.")
    parser.add_argument("--models", nargs="*", default=["all"], help="Models to run: all, medswin, bge-gemma, bge-m3")
    parser.add_argument("--run-name", default=None, help="Output run name under <root>/outputs")
    parser.add_argument("--download-only", action="store_true", help="Download selected models and exit")
    parser.add_argument("--skip-download", action="store_true", help="Do not download missing models before benchmarking")
    parser.add_argument("--force-download", action="store_true", help="Force re-download model snapshots")
    parser.add_argument("--local-files-only", action="store_true", help="Never access Hugging Face Hub; require local model folders")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision/commit for all model downloads")
    parser.add_argument("--hf-token", default=None, help="HF token string, or 'true' to use logged-in token. Avoid putting tokens in shell history.")
    parser.add_argument("--hf-transfer", action="store_true", help="Enable hf_transfer if the package is installed")

    task_group = parser.add_argument_group("MTEB task selection")
    task_group.add_argument("--benchmark", default=None, help='Optional benchmark, e.g. "MTEB(eng, v2)"')
    task_group.add_argument("--tasks", nargs="*", default=None, help="Specific reranking task names, comma/space separated")
    task_group.add_argument("--languages", nargs="*", default=None, help='Language filters, e.g. "eng"')
    task_group.add_argument("--domains", nargs="*", default=None, help='Domain filters, e.g. "Medical" "Scientific"')
    task_group.add_argument("--medical-only", action="store_true", help='Shortcut for --domains Medical')
    task_group.add_argument("--eval-splits", nargs="*", default=None, help='Splits to evaluate, e.g. "test"')
    task_group.add_argument("--text-only", action=argparse.BooleanOptionalAction, default=True, help="Restrict to text tasks")
    task_group.add_argument("--limit-tasks", type=int, default=None, help="Limit number of tasks for smoke testing")
    task_group.add_argument("--list-tasks", action="store_true", help="Print selected tasks and exit")

    infer_group = parser.add_argument_group("Inference")
    infer_group.add_argument("--implementation", default="transformers", choices=["transformers", "flagembedding"], help="Inference implementation. Use transformers to avoid FlagEmbedding prepare_for_model tokenizer errors.")
    infer_group.add_argument("--batch-size", type=int, default=None, help="Override all per-model batch sizes")
    infer_group.add_argument("--max-length", type=int, default=None, help="Override all per-model max lengths")
    infer_group.add_argument("--use-fp16", action=argparse.BooleanOptionalAction, default=True, help="Use fp16 in FlagEmbedding")
    infer_group.add_argument("--use-bf16", action="store_true", help="Use bf16 instead of fp16 when supported")
    infer_group.add_argument("--normalize-scores", action=argparse.BooleanOptionalAction, default=False, help="Normalize reranker scores via backend sigmoid when supported")
    infer_group.add_argument("--device", default="cuda:0", help='Single device, e.g. "cuda:0" or "cpu"')
    infer_group.add_argument("--devices", nargs="*", default=None, help='Multiple devices, e.g. "cuda:0,cuda:1"')
    infer_group.add_argument("--query-prefix", default="", help="Optional prefix prepended to every query")
    infer_group.add_argument("--document-prefix", default="", help="Optional prefix prepended to every document")
    infer_group.add_argument("--llm-prompt", default=None, help="Prompt used by Gemma-style LLM rerankers. Defaults to the BGE model-card prompt.")

    eval_group = parser.add_argument_group("Evaluation and outputs")
    eval_group.add_argument("--save-predictions", action="store_true", help="Save MTEB per-query predictions when supported")
    eval_group.add_argument("--overwrite-strategy", default="only-missing", choices=["always", "never", "only-missing", "only-cache"], help="MTEB cache overwrite policy")
    eval_group.add_argument("--no-cache", action="store_true", help="Disable MTEB result cache")
    eval_group.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True, help="Continue when an individual task fails")
    eval_group.add_argument("--num-proc", type=int, default=None, help="Processes for MTEB data loading/transforms")
    eval_group.add_argument("--public-only", action=argparse.BooleanOptionalAction, default=True, help="Run public tasks only")
    eval_group.add_argument("--co2-tracker", action=argparse.BooleanOptionalAction, default=False, help="Enable MTEB codecarbon tracking if installed")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    root = Path(args.root).expanduser().resolve() if args.root else project_root_from_script()
    root.mkdir(parents=True, exist_ok=True)
    set_repro_and_perf_env(root, args.hf_transfer)

    specs = parse_models(args.models)
    print(f"Project root: {root}")
    print("Selected models:")
    for spec in specs:
        print(f"  - {spec.display_name}: {spec.repo_id} -> {model_local_dir(root, spec)}")
    print_cuda_summary()

    if args.skip_download:
        local_dirs = {spec.key: model_local_dir(root, spec) for spec in specs}
        missing = [spec for spec in specs if not has_model_files(local_dirs[spec.key])]
        if missing:
            missing_text = ", ".join(f"{m.display_name} at {local_dirs[m.key]}" for m in missing)
            raise SystemExit(f"Missing local model files and --skip-download was used: {missing_text}")
    else:
        local_dirs = download_selected_models(args, specs, root)

    if args.download_only:
        print("\nDownloaded/verified model directories:")
        for spec in specs:
            print(f"  - {spec.display_name}: {local_dirs[spec.key]}")
        return 0

    tasks = build_tasks(args)
    if not tasks:
        raise SystemExit(
            "No reranking tasks matched your filters. Try removing --medical-only/--domains, "
            "or use --benchmark \"MTEB(eng, v2)\"."
        )

    run_name = safe_name(args.run_name or now_run_name())
    run_dir = root / "outputs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = task_manifest(tasks)
    write_json(run_dir / "selected_tasks.json", manifest)
    write_json(
        run_dir / "run_config.json",
        {
            "root": str(root),
            "models": [asdict(spec) | {"local_dir": str(local_dirs[spec.key])} for spec in specs],
            "tasks": manifest,
            "args": sanitize_args(vars(args)),
        },
    )

    print(f"\nSelected {len(tasks)} MTEB reranking task(s). Manifest: {run_dir / 'selected_tasks.json'}")
    for item in manifest:
        print(f"  - {item['name']} | domains={item['domains']} | languages={item['languages']}")

    if args.list_tasks:
        return 0

    all_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    started = time.time()

    for spec in specs:
        rows, failures = run_one_model(spec, local_dirs[spec.key], tasks, args, root, run_dir)
        all_rows.extend(rows)
        failed_rows.extend(failures)

    elapsed = time.time() - started
    write_cross_model_outputs(run_dir, all_rows, failed_rows)
    write_json(
        run_dir / "finished.json",
        {
            "elapsed_seconds": elapsed,
            "num_models": len(specs),
            "num_tasks": len(tasks),
            "num_metric_rows": len(all_rows),
            "num_failed_tasks": len(failed_rows),
        },
    )

    print(f"\nDone. Outputs written to: {run_dir}")
    print("Main output files:")
    print(f"  - {run_dir / 'run_config.json'}")
    print(f"  - {run_dir / 'selected_tasks.json'}")
    print(f"  - {run_dir / 'all_models_metrics_long.csv'}")
    print(f"  - {run_dir / 'comparison_by_task_metric.csv'}")
    print(f"  - {run_dir / 'summary_by_model_metric.csv'}")
    print(f"  - {run_dir / 'model_run_summary.csv'}")
    print(f"  - {run_dir / 'failed_tasks.json'}")
    print("Per-model outputs are under:")
    print(f"  - {run_dir / 'models'}")
    if failed_rows:
        print(f"\nCompleted with {len(failed_rows)} failed model/task run(s). Check failed_tasks.json.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
