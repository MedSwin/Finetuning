#!/usr/bin/env python3
"""
Preprocess or repair HealthBench for benchmarking.

Modes:
- preprocess: transform raw HealthBench rows into English plaintext fields.
- fix: audit an existing processed JSONL, detect rows that are still unsuitable,
       and re-distill only the affected fields.

python fix_healthbench.py \
  --mode fix \
  --input healthbench_processed.jsonl \
  --original-input healthbench.jsonl \
  --output healthbench_processed_fixed.jsonl \
  --audit-report healthbench_audit_report.json \
  --cache healthbench_cache.jsonl \
  --llm-log healthbench_fix_llm_logs.jsonl \
  --deployment gpt-5-nano \
  --overwrite

The fix mode is designed for datasets that are already structurally close to
benchmark-ready, but still contain issues such as untranslated text, markdown,
placeholders, HTML, or malformed benchmark-driving fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm
API_VERSION = "2024-12-01-preview"
DEFAULT_DEPLOYMENT = "gpt-5-nano"
DEFAULT_CHECKPOINT_INTERVAL = 10


SYSTEM_PROMPT = """You are a medical data extraction and translation engine.

Your task is to convert the input field into a concise, English plaintext summary.

Strict Requirements:
1. LANGUAGE: If the input is not English, translate it faithfully to English.
2. CONTENT: Preserve every medical fact, vital sign, date, medication, and clinical uncertainty.
3. STRIP: Remove all greetings, apologies, conversational filler, markdown, tables, and HTML.
4. COMPRESSION: If the text is long, summarize it to {min_words}-{max_words} words. If it is already shorter than {max_words} words, do not add any text.
5. FORMAT: Return ONLY the raw plaintext. No labels ("Summary:", "Translation:"), no preamble, and no markdown formatting.
6. INTEGRITY: Never hallucinate or omit clinical details. No commentary or reasoning.
"""


USER_TEMPLATE = """Process this {field_type} field into clinical plaintext:

{text}
"""


SCRIPT_RE = re.compile(
    r"[\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF"
    r"\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F"
    r"\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0E00-\u0E7F"
    r"\u10A0-\u10FF\u1100-\u11FF\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF"
    r"\uAC00-\uD7AF]"
)
INVERTED_PUNCT_RE = re.compile(r"[¿¡]")
HTML_RE = re.compile(r"<[^>]+>")
MARKDOWN_RE = re.compile(r"(^|\n)\s{0,3}(#{1,6}\s|[-*+]\s|\d+\.\s)|\|.+\||```", re.M)
LABEL_RE = re.compile(
    r"^\s*(summary|translation|processed|output|answer|final answer|clinical plaintext)\s*:\s*",
    re.I,
)
ROLE_RE = re.compile(r"^\s*(user|assistant|system)\s*:\s*", re.I)
PLACEHOLDER_RE = re.compile(
    r"\b(template|placeholder|lorem ipsum|insert here|tbd|to be added|fill in|dummy text)\b",
    re.I,
)

FOREIGN_STEMS = {
    "pt": [
        " você ", " nao ", " não ", " para ", " uma ", " tomar ", " dor ",
        " médico ", " febre ", " garganta ", " ajuda ", " remédio ", " dias ",
    ],
    "es": [
        " usted ", " para ", " cómo ", " como ", " dolor ", " días ", " médico ",
        " fiebre ", " garganta ", " ayuda ", " hola ", " puedo ",
    ],
    "fr": [
        " vous ", " comment ", " douleur ", " médecin ", " fièvre ", " bonjour ",
        " merci ", " est-ce ", " jours ",
    ],
    "de": [
        " ich ", " nicht ", " schmerzen ", " arzt ", " fieber ", " bitte ",
    ],
    "it": [
        " dolore ", " medico ", " posso ", " giorni ",
    ],
    "ru": [
        " у меня ", " пожалуйста ", " врач ", " боль ", " как ",
    ],
    "ar": [
        " هذا ", " ألم ", " طبيب ",
    ],
    "hi": [
        " क्या ", " मुझे ", " दर्द ", " डॉक्टर ",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess or repair HealthBench with Azure OpenAI.")
    parser.add_argument("--mode", choices=["preprocess", "fix"], default="preprocess")
    parser.add_argument("--input", required=True, help="Path to input JSONL.")
    parser.add_argument("--output", required=True, help="Path to output JSONL.")
    parser.add_argument(
        "--cache",
        default=None,
        help="Optional JSONL cache file for field-level results. Recommended for resumability.",
    )
    parser.add_argument("--deployment", default=DEFAULT_DEPLOYMENT, help="Azure deployment/model name.")
    parser.add_argument("--api-version", default=API_VERSION, help="Azure OpenAI API version.")
    parser.add_argument("--max-completion-tokens", type=int, default=1200)
    parser.add_argument("--min-output-words", type=int, default=75)
    parser.add_argument("--max-output-words", type=int, default=150)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--llm-log",
        default=None,
        help="Optional JSONL file to log every LLM request/response/error.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists (only if skip-rebuild is False).",
    )
    parser.add_argument(
        "--replace-original-fields",
        action="store_true",
        help="In preprocess mode, replace prompt and ideal_completion with processed text.",
    )
    parser.add_argument(
        "--skip-rebuild",
        action="store_true",
        default=True,
        help="Skip records already found in the output file (default: True).",
    )
    parser.add_argument(
        "--no-skip-rebuild",
        action="store_false",
        dest="skip_rebuild",
        help="Disable skipping and rebuild everything.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help="Rows to buffer before appending to output.",
    )
    parser.add_argument(
        "--audit-report",
        default=None,
        help="Optional JSON path to save audit summary in fix mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="In fix mode, only audit and report. Do not write repaired JSONL or call the LLM.",
    )
    parser.add_argument(
        "--fix-max-rows",
        type=int,
        default=0,
        help="In fix mode, cap how many bad rows are reprocessed. 0 = all flagged rows.",
    )
    parser.add_argument(
        "--original-input",
        default=None,
        help="Optional raw/original HealthBench JSONL. In fix mode, flagged rows will be re-distilled from the original row matched by prompt_id when available.",
    )
    return parser.parse_args()


def build_client(endpoint: str, api_key: str, api_version: str):
    from openai import AzureOpenAI
    return AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=api_key)


def build_client_if_needed(args: argparse.Namespace, needs_llm: bool) -> Tuple[Optional[object], Optional[str]]:
    if not needs_llm:
        return None, None
    
    # Try to get from environment, fallback to your provided credentials
    endpoint = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
    api_key = os.getenv("AZURE_AI_FOUNDRY_API_KEY")
    
    if not endpoint or not api_key:
        raise RuntimeError("Missing Azure credentials.")
        
    client = build_client(endpoint=endpoint, api_key=api_key, api_version=args.api_version)
    return client, endpoint


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_num}: {e}") from e


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]], mode: str = "w") -> None:
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_whitespace(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_markdown_fences(text: str) -> str:
    text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text.strip())
    text = re.sub(r"\n```$", "", text.strip())
    return text.strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join([content_to_text(x) for x in value if content_to_text(x)])
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if "content" in value:
            return content_to_text(value["content"])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def prompt_to_plaintext(prompt_value: Any) -> str:
    if prompt_value is None:
        return ""
    if isinstance(prompt_value, str):
        return normalize_whitespace(prompt_value)
    if isinstance(prompt_value, list):
        lines: List[str] = []
        for msg in prompt_value:
            if isinstance(msg, dict):
                role = str(msg.get("role", "user")).strip() or "user"
                content_text = normalize_whitespace(content_to_text(msg.get("content", "")))
                if content_text:
                    lines.append(f"{role}: {content_text}")
            else:
                text = normalize_whitespace(content_to_text(msg))
                if text:
                    lines.append(text)
        return "\n".join(lines).strip()
    if isinstance(prompt_value, dict):
        return normalize_whitespace(content_to_text(prompt_value))
    return normalize_whitespace(str(prompt_value))


class JsonlCache:
    def __init__(self, path: Optional[str]):
        self.path = path
        self.data: Dict[str, str] = {}
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        self.data[row["key"]] = row["value"]
                    except Exception:
                        continue

    def get(self, key: str) -> Optional[str]:
        return self.data.get(key)

    def set(self, key: str, value: str) -> None:
        self.data[key] = value
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


class AzureFieldProcessor:
    def __init__(
        self,
        client: object,
        deployment: str,
        min_words: int,
        max_words: int,
        max_completion_tokens: int,
        max_retries: int,
        retry_sleep: float,
        cache: Optional[JsonlCache] = None,
        log_path: Optional[str] = None,
    ) -> None:
        self.client = client
        self.deployment = deployment
        self.min_words = min_words
        self.max_words = max_words
        self.max_completion_tokens = max_completion_tokens
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.cache = cache or JsonlCache(None)
        self.log_path = log_path

    def _append_log(self, payload: Dict[str, Any]) -> None:
        if not self.log_path:
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def process_field(self, field_type: str, text: str) -> str:
        text = normalize_whitespace(text)
        if not text:
            return ""

        text_hash = sha256_text(text)
        cache_key = f"{field_type}:{self.min_words}:{self.max_words}:{text_hash}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            self._append_log(
                {
                    "event": "cache_hit",
                    "field_type": field_type,
                    "cache_key": cache_key,
                    "text_sha256": text_hash,
                    "cached_output": cached,
                    "timestamp": time.time(),
                }
            )
            return cached

        system_prompt = SYSTEM_PROMPT.format(min_words=self.min_words, max_words=self.max_words)
        user_prompt = USER_TEMPLATE.format(field_type=field_type, text=text)

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            request_id = hashlib.sha256(
                f"{field_type}|{attempt}|{time.time()}|{cache_key}".encode("utf-8")
            ).hexdigest()[:16]

            request_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            self._append_log(
                {
                    "event": "llm_request",
                    "request_id": request_id,
                    "field_type": field_type,
                    "attempt": attempt,
                    "deployment": self.deployment,
                    "api_version": getattr(self.client, "api_version", API_VERSION),
                    "max_completion_tokens": self.max_completion_tokens,
                    "cache_key": cache_key,
                    "text_sha256": text_hash,
                    "messages": request_messages,
                    "timestamp": time.time(),
                }
            )

            try:
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=request_messages,
                    max_completion_tokens=self.max_completion_tokens,
                )
                raw_out = response.choices[0].message.content or ""
                out = normalize_whitespace(strip_markdown_fences(raw_out))

                usage = None
                if getattr(response, "usage", None) is not None:
                    if hasattr(response.usage, "model_dump"):
                        usage = response.usage.model_dump()
                    elif isinstance(response.usage, dict):
                        usage = dict(response.usage)

                self._append_log(
                    {
                        "event": "llm_response",
                        "request_id": request_id,
                        "field_type": field_type,
                        "attempt": attempt,
                        "cache_key": cache_key,
                        "text_sha256": text_hash,
                        "response_text": raw_out,
                        "normalized_response_text": out,
                        "response_model": getattr(response, "model", None),
                        "finish_reason": response.choices[0].finish_reason if getattr(response, "choices", None) else None,
                        "usage": usage,
                        "timestamp": time.time(),
                    }
                )
                self.cache.set(cache_key, out)
                return out
            except Exception as e:
                last_err = e
                self._append_log(
                    {
                        "event": "llm_error",
                        "request_id": request_id,
                        "field_type": field_type,
                        "attempt": attempt,
                        "cache_key": cache_key,
                        "text_sha256": text_hash,
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep * attempt)
                else:
                    break

        raise RuntimeError(f"Azure request failed after {self.max_retries} attempts: {last_err}")


def get_ideal_completion(record: Dict[str, Any]) -> str:
    icd = record.get("ideal_completions_data")
    if isinstance(icd, dict):
        return content_to_text(icd.get("ideal_completion", ""))
    return ""


def set_ideal_completion(record: Dict[str, Any], value: str) -> None:
    if not isinstance(record.get("ideal_completions_data"), dict):
        record["ideal_completions_data"] = {}
    record["ideal_completions_data"]["ideal_completion"] = value


def process_record(
    record: Dict[str, Any],
    processor: AzureFieldProcessor,
    replace_original_fields: bool = False,
) -> Dict[str, Any]:
    prompt_raw = prompt_to_plaintext(record.get("prompt"))
    ideal_raw = normalize_whitespace(get_ideal_completion(record))
    prompt_processed = processor.process_field("prompt", prompt_raw)
    ideal_processed = processor.process_field("ideal_completion", ideal_raw)

    out = dict(record)
    if replace_original_fields:
        out["prompt"] = [{"role": "user", "content": prompt_processed}]
        set_ideal_completion(out, ideal_processed)
        out.pop("processed_prompt_en_plaintext", None)
        out.pop("processed_ideal_completion_en_plaintext", None)
    else:
        out["processed_prompt_en_plaintext"] = prompt_processed
        out["processed_ideal_completion_en_plaintext"] = ideal_processed

    meta = out.get("preprocessing_meta")
    if not isinstance(meta, dict):
        meta = {}
    meta.update(
        {
            "processor": "azure_gpt5nano_single_field",
            "api_version": getattr(processor.client, "api_version", API_VERSION),
            "prompt_output_style": "english_plaintext",
            "ideal_completion_output_style": "english_plaintext",
            "single_field_requests_only": True,
            "target_word_range_when_compressed": [processor.min_words, processor.max_words],
            "replace_original_fields": replace_original_fields,
        }
    )
    out["preprocessing_meta"] = meta
    return out


def norm_text(s: str) -> str:
    return " ".join((s or "").split()).strip()


def benchmark_content_to_text(value: Any) -> str:
    return content_to_text(value)


def extract_reference_for_benchmark(obj: Dict[str, Any]) -> Optional[str]:
    ideal = obj.get("ideal_completions_data") or {}
    primary = norm_text(benchmark_content_to_text(ideal.get("ideal_completion")))
    if primary:
        return primary
    for x in ideal.get("ideal_completions_ref_completions") or []:
        x = norm_text(benchmark_content_to_text(x))
        if x:
            return x
    for key in [
        "processed_ideal_completion_en_plaintext",
        "ideal_completion",
        "answer",
        "reference",
        "gold",
        "gold_answer",
    ]:
        val = norm_text(benchmark_content_to_text(obj.get(key)))
        if val:
            return val
    return None


def extract_messages_for_benchmark(obj: Dict[str, Any]) -> List[Dict[str, str]]:
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
            content = norm_text(benchmark_content_to_text(raw.get("content", raw)))
            if content:
                return [{"role": role, "content": content}]
            continue
        if isinstance(raw, list):
            for m in raw:
                if isinstance(m, dict):
                    role = str(m.get("role", "user")).strip().lower() or "user"
                    content = norm_text(benchmark_content_to_text(m.get("content", "")))
                    if content:
                        cleaned.append({"role": role, "content": content})
                else:
                    content = norm_text(benchmark_content_to_text(m))
                    if content:
                        cleaned.append({"role": "user", "content": content})
            if cleaned:
                return cleaned
    return []


def token_hits_lower(text_lower: str) -> List[Tuple[str, str]]:
    hits: List[Tuple[str, str]] = []
    for lang, stems in FOREIGN_STEMS.items():
        for stem in stems:
            if stem in text_lower:
                hits.append((lang, stem.strip()))
    return hits


def looks_non_english(text: str) -> bool:
    text_lower = f" {text.lower()} "
    if not text_lower.strip():
        return False
    if SCRIPT_RE.search(text_lower):
        return True
    if INVERTED_PUNCT_RE.search(text_lower):
        return True
    hits = token_hits_lower(text_lower)
    if len(hits) >= 2:
        return True
    if re.search(r"\b(dolor|douleur|schmerzen|dor|médico|medico|bonjour|salut|gracias|obrigado|obrigada|merci|você|usted)\b", text_lower):
        return True
    return False


def audit_text(text: Any, field_name: str, max_words: int) -> List[str]:
    issues: List[str] = []
    if not isinstance(text, str):
        issues.append(f"{field_name}_not_string")
        text = "" if text is None else str(text)
    t = text.strip()
    if not t:
        issues.append(f"{field_name}_empty")
        return issues
    if LABEL_RE.search(t):
        issues.append(f"{field_name}_label_prefix")
    if ROLE_RE.search(t):
        issues.append(f"{field_name}_role_prefix")
    if HTML_RE.search(t):
        issues.append(f"{field_name}_html")
    if MARKDOWN_RE.search(t):
        issues.append(f"{field_name}_markdown")
    if PLACEHOLDER_RE.search(t):
        issues.append(f"{field_name}_placeholder")
    if looks_non_english(t):
        issues.append(f"{field_name}_non_english")
    if len(t.split()) > max_words:
        issues.append(f"{field_name}_over_max_words")
    return issues


def audit_record(record: Dict[str, Any], max_words: int) -> Dict[str, Any]:
    prompt_issues: List[str] = []
    ideal_issues: List[str] = []
    bench_issues: List[str] = []

    prompt_value = record.get("prompt")
    prompt_text = ""
    if not isinstance(prompt_value, list) or len(prompt_value) != 1:
        prompt_issues.append("prompt_structure_not_single_message")
        prompt_text = prompt_to_plaintext(prompt_value)
    else:
        msg = prompt_value[0]
        if not isinstance(msg, dict):
            prompt_issues.append("prompt_message_not_dict")
            prompt_text = prompt_to_plaintext(prompt_value)
        else:
            if msg.get("role") != "user":
                prompt_issues.append("prompt_role_not_user")
            prompt_text = msg.get("content", "") if isinstance(msg.get("content"), str) else content_to_text(msg.get("content"))

    ideal_text = get_ideal_completion(record)
    prompt_issues.extend(audit_text(prompt_text, "prompt", max_words=max_words))
    ideal_issues.extend(audit_text(ideal_text, "ideal", max_words=max_words))

    if not extract_reference_for_benchmark(record):
        bench_issues.append("benchmark_no_reference")
    if not extract_messages_for_benchmark(record):
        bench_issues.append("benchmark_no_prompt")

    meta = record.get("preprocessing_meta") if isinstance(record.get("preprocessing_meta"), dict) else {}
    if meta.get("unresolved_prompt_translation") is True:
        prompt_issues.append("meta_unresolved_prompt_translation")
    if meta.get("unresolved_ideal_translation") is True:
        ideal_issues.append("meta_unresolved_ideal_translation")

    prompt_issues = sorted(set(prompt_issues))
    ideal_issues = sorted(set(ideal_issues))
    bench_issues = sorted(set(bench_issues))
    needs_fix = bool(prompt_issues or ideal_issues or bench_issues)

    return {
        "prompt_id": record.get("prompt_id"),
        "prompt_issues": prompt_issues,
        "ideal_issues": ideal_issues,
        "bench_issues": bench_issues,
        "needs_fix": needs_fix,
        "fix_prompt": bool(prompt_issues or bench_issues),
        "fix_ideal": bool(ideal_issues or bench_issues),
    }


def update_fix_metadata(
    out: Dict[str, Any],
    before: Dict[str, Any],
    after: Dict[str, Any],
    processor: Optional[AzureFieldProcessor],
    prompt_fixed: bool,
    ideal_fixed: bool,
) -> None:
    meta = out.get("preprocessing_meta")
    if not isinstance(meta, dict):
        meta = {}
    meta.update(
        {
            "processor": "azure_gpt5nano_single_field_fix_mode",
            "api_version": getattr(getattr(processor, "client", None), "api_version", API_VERSION),
            "prompt_output_style": "english_plaintext",
            "ideal_completion_output_style": "english_plaintext",
            "single_user_prompt_message": True,
            "target_word_range_when_compressed": [processor.min_words, processor.max_words] if processor else meta.get("target_word_range_when_compressed", [75, 150]),
            "fix_mode_applied": True,
            "fix_prompt": prompt_fixed,
            "fix_ideal_completion": ideal_fixed,
            "fix_prompt_issues_before": before["prompt_issues"],
            "fix_ideal_issues_before": before["ideal_issues"],
            "fix_bench_issues_before": before["bench_issues"],
            "fix_prompt_issues_after": after["prompt_issues"],
            "fix_ideal_issues_after": after["ideal_issues"],
            "fix_bench_issues_after": after["bench_issues"],
            "benchmark_ready_after_fix": not after["needs_fix"],
            "unresolved_prompt_translation": "prompt_non_english" in after["prompt_issues"],
            "unresolved_ideal_translation": "ideal_non_english" in after["ideal_issues"],
        }
    )
    out["preprocessing_meta"] = meta


def build_audit_summary(audits: List[Dict[str, Any]], total_rows: int) -> Dict[str, Any]:
    issue_counts = Counter()
    prompt_fix = 0
    ideal_fix = 0
    bench_fix = 0
    rows_needing_fix = 0
    samples: List[Dict[str, Any]] = []

    for audit in audits:
        if audit["needs_fix"]:
            rows_needing_fix += 1
        if audit["fix_prompt"]:
            prompt_fix += 1
        if audit["fix_ideal"]:
            ideal_fix += 1
        if audit["bench_issues"]:
            bench_fix += 1
        for issue in audit["prompt_issues"] + audit["ideal_issues"] + audit["bench_issues"]:
            issue_counts[issue] += 1
        if audit["needs_fix"] and len(samples) < 25:
            samples.append(
                {
                    "prompt_id": audit.get("prompt_id"),
                    "prompt_issues": audit["prompt_issues"],
                    "ideal_issues": audit["ideal_issues"],
                    "bench_issues": audit["bench_issues"],
                }
            )

    return {
        "total_rows": total_rows,
        "rows_needing_fix": rows_needing_fix,
        "rows_needing_fix_pct": round((rows_needing_fix / total_rows) * 100, 2) if total_rows else 0.0,
        "rows_fix_prompt": prompt_fix,
        "rows_fix_ideal_completion": ideal_fix,
        "rows_fix_benchmark_structure": bench_fix,
        "issue_counts": dict(sorted(issue_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "sample_flagged_rows": samples,
    }


def load_existing_ids(path: str) -> set:
    existing_ids = set()
    if os.path.exists(path):
        for row in iter_jsonl(path):
            if "prompt_id" in row:
                existing_ids.add(row["prompt_id"])
    return existing_ids




def load_record_map_by_prompt_id(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    mapping: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(iter_jsonl(path), start=1):
        prompt_id = row.get("prompt_id", f"row_{idx}")
        mapping[prompt_id] = row
    return mapping


def run_preprocess_mode(args: argparse.Namespace) -> int:
    if os.path.exists(args.output) and not args.overwrite and not args.skip_rebuild:
        print(f"Output already exists: {args.output}. Use --overwrite or --skip-rebuild.", file=sys.stderr)
        return 2

    client, _ = build_client_if_needed(args, needs_llm=True)
    cache = JsonlCache(args.cache)
    processor = AzureFieldProcessor(
        client=client,
        deployment=args.deployment,
        min_words=args.min_output_words,
        max_words=args.max_output_words,
        max_completion_tokens=args.max_completion_tokens,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        cache=cache,
        log_path=args.llm_log,
    )

    existing_ids = load_existing_ids(args.output) if os.path.exists(args.output) else set()
    total_records = sum(1 for _ in open(args.input, "r", encoding="utf-8"))

    total_new_processed = 0
    skipped_records: List[Dict[str, str]] = []
    checkpoint_buffer: List[Dict[str, Any]] = []

    try:
        with tqdm(total=total_records, desc="Processing HealthBench", unit="row") as pbar:
            for idx, record in enumerate(iter_jsonl(args.input), start=1):
                prompt_id = record.get("prompt_id", f"row_{idx}")
                if prompt_id in existing_ids:
                    pbar.update(1)
                    continue
                try:
                    processed = process_record(
                        record=record,
                        processor=processor,
                        replace_original_fields=args.replace_original_fields,
                    )
                    checkpoint_buffer.append(processed)
                    total_new_processed += 1
                    if len(checkpoint_buffer) >= args.checkpoint_interval:
                        write_jsonl(args.output, checkpoint_buffer, mode="a")
                        checkpoint_buffer = []
                    pbar.set_postfix({"new": total_new_processed, "err": len(skipped_records)})
                    pbar.update(1)
                except Exception as e:
                    error_msg = str(e).split("\n")[0]
                    tqdm.write(f"Skipping {prompt_id} due to error: {error_msg}")
                    skipped_records.append({"id": prompt_id, "error": error_msg})
                    pbar.update(1)
    except KeyboardInterrupt:
        print("\n[!] Interrupted. Saving progress...", file=sys.stderr)
    finally:
        if checkpoint_buffer:
            write_jsonl(args.output, checkpoint_buffer, mode="a")
            print(f"Final flush: Saved {len(checkpoint_buffer)} rows.", file=sys.stderr)

    print("-" * 30)
    print("Run Complete.")
    print(f" - New rows added: {total_new_processed}")
    print(f" - Skipped (errors): {len(skipped_records)}")
    print(f" - Output file: {args.output}")
    print("-" * 30)
    return 0


def run_fix_mode(args: argparse.Namespace) -> int:
    if os.path.exists(args.output) and not args.overwrite and not args.skip_rebuild and not args.dry_run:
        print(f"Output already exists: {args.output}. Use --overwrite or --skip-rebuild.", file=sys.stderr)
        return 2

    original_map = load_record_map_by_prompt_id(args.original_input)
    existing_ids = load_existing_ids(args.output) if os.path.exists(args.output) else set()
    
    # Setup LLM Processor
    client, _ = build_client_if_needed(args, needs_llm=not args.dry_run)
    processor = AzureFieldProcessor(
        client=client, deployment=args.deployment,
        min_words=args.min_output_words, max_words=args.max_output_words,
        max_completion_tokens=args.max_completion_tokens,
        max_retries=args.max_retries, retry_sleep=args.retry_sleep,
        cache=JsonlCache(args.cache), log_path=args.llm_log,
    )

    checkpoint_buffer, output_rows, errors, all_before_audits = [], [], [], []
    repaired, skipped_clean, total_seen = 0, 0, 0

    try:
        # We use a simple progress bar without pre-counting the whole file for speed
        with tqdm(desc="Fixing HealthBench", unit="row") as pbar:
            for idx, record in enumerate(iter_jsonl(args.input), start=1):
                total_seen += 1
                prompt_id = record.get("prompt_id", f"row_{idx}")
                if prompt_id in existing_ids:
                    pbar.update(1)
                    continue

                # Detect directly if fix is needed
                audit_before = audit_record(record, max_words=args.max_output_words)
                all_before_audits.append(audit_before)

                if not audit_before["needs_fix"]:
                    skipped_clean += 1
                    pbar.update(1)
                    continue 

                if args.dry_run:
                    repaired += 1
                    pbar.update(1)
                    continue

                if args.fix_max_rows > 0 and repaired >= args.fix_max_rows:
                    break

                try:
                    out = deepcopy(record)
                    source_record = original_map.get(prompt_id, record)
                    prompt_fixed, ideal_fixed = False, False

                    if audit_before["fix_prompt"]:
                        prompt_raw = prompt_to_plaintext(source_record.get("prompt") or record.get("prompt"))
                        prompt_clean = processor.process_field("prompt", prompt_raw)
                        out["prompt"] = [{"role": "user", "content": prompt_clean}]
                        out.pop("processed_prompt_en_plaintext", None)
                        prompt_fixed = True

                    if audit_before["fix_ideal"]:
                        ideal_raw = get_ideal_completion(source_record) or get_ideal_completion(record)
                        ideal_clean = processor.process_field("ideal_completion", ideal_raw)
                        set_ideal_completion(out, ideal_clean)
                        out.pop("processed_ideal_completion_en_plaintext", None)
                        ideal_fixed = True

                    audit_after = audit_record(out, max_words=args.max_output_words)
                    update_fix_metadata(out, audit_before, audit_after, processor, prompt_fixed, ideal_fixed)
                    
                    checkpoint_buffer.append(out)
                    repaired += 1
                    if len(checkpoint_buffer) >= args.checkpoint_interval:
                        write_jsonl(args.output, checkpoint_buffer, mode="a")
                        output_rows.extend(checkpoint_buffer)
                        checkpoint_buffer = []
                    
                    pbar.set_postfix({"repaired": repaired, "skipped": skipped_clean, "err": len(errors)})
                    pbar.update(1)
                except Exception as e:
                    errors.append({"id": prompt_id, "error": str(e).split("\n")[0]})
                    pbar.update(1)
                    
    except KeyboardInterrupt:
        print("\n[!] Interrupted.", file=sys.stderr)
    finally:
        if checkpoint_buffer:
            write_jsonl(args.output, checkpoint_buffer, mode="a")
            output_rows.extend(checkpoint_buffer)

    summary_before = build_audit_summary(all_before_audits, total_rows=total_seen)
    summary_after = build_audit_summary([audit_record(r, args.max_output_words) for r in output_rows], len(output_rows))

    final_report = {
        "mode": "fix",
        "summary": {
            "total_input_scanned": total_seen,
            "repaired_count": repaired,
            "skipped_clean_count": skipped_clean,
            "error_count": len(errors)
        },
        "audit_before": summary_before,
        "audit_after": summary_after,
        "errors": errors,
        "output": args.output,
    }
    if args.audit_report:
        with open(args.audit_report, "w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)

    print(json.dumps(final_report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


def main() -> int:
    args = parse_args()
    if args.mode == "preprocess":
        return run_preprocess_mode(args)
    return run_fix_mode(args)


if __name__ == "__main__":
    sys.exit(main())
