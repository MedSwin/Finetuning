#!/usr/bin/env python3
"""
Preprocess HealthBench with Azure OpenAI

What this script does per example:
1) Extracts the prompt conversation into plain text.
2) Detects whether the prompt text is non-English.
3) If non-English, translates it into English while preserving all facts and context.
4) Converts markdown / tables / formatting into simple plain text.
5) If content is too long, compresses it to roughly 75-150 words while preserving facts.
6) Repeats the same process for ideal_completion.

Important:
- Each LLM request handles exactly ONE field at a time:
  either the prompt OR the ideal_completion.
- No batching is used for field transformation requests.
- By default, the original dataset is preserved and processed fields are added.
- If --replace-original-fields is used, the script overwrites prompt and
  ideal_completions_data['ideal_completion'] with the processed English plaintext
  and does NOT add duplicate processed_* fields.

Expected input format:
- JSONL file where each row contains:
  - prompt: typically a list of chat messages with {role, content}
  - ideal_completions_data.ideal_completion: target answer text

Environment variables:
- AZURE_AI_FOUNDRY_ENDPOINT
- AZURE_AI_FOUNDRY_API_KEY

Example:
python3 prep_healthbench.py \
  --input healthbench.jsonl \
  --output healthbench_processed.jsonl \
  --cache healthbench_cache.jsonl \
  --llm-log healthbench_llm_logs.jsonl \
  --deployment gpt-5-nano \
  --replace-original-fields
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Optional

from openai import AzureOpenAI


API_VERSION = "2024-12-01-preview"
DEFAULT_DEPLOYMENT = "gpt-5-nano"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess HealthBench with Azure OpenAI.")
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
        help="Replace prompt and ideal_completion with processed text.",
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
    return parser.parse_args()


def build_client(endpoint: str, api_key: str, api_version: str) -> AzureOpenAI:
    return AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=api_key)


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


def prompt_to_plaintext(prompt_value: Any) -> str:
    """
    Convert HealthBench prompt field to a minimal plain text representation.
    Usually prompt is a list of {role, content} messages.
    """
    if prompt_value is None:
        return ""

    if isinstance(prompt_value, str):
        return normalize_whitespace(prompt_value)

    if isinstance(prompt_value, list):
        lines: List[str] = []
        for msg in prompt_value:
            if isinstance(msg, dict):
                role = str(msg.get("role", "user")).strip() or "user"
                content = msg.get("content", "")
                content_text = content_to_text(content)
                content_text = normalize_whitespace(content_text)
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


def content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            parts.append(content_to_text(item))
        return "\n".join([p for p in parts if p])
    if isinstance(value, dict):
        if "text" in value and isinstance(value["text"], str):
            return value["text"]
        if "content" in value:
            return content_to_text(value["content"])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        client: AzureOpenAI,
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
                        "finish_reason": (
                            response.choices[0].finish_reason
                            if getattr(response, "choices", None) else None
                        ),
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


def strip_markdown_fences(text: str) -> str:
    text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text.strip())
    text = re.sub(r"\n```$", "", text.strip())
    return text.strip()


def get_ideal_completion(record: Dict[str, Any]) -> str:
    icd = record.get("ideal_completions_data")
    if isinstance(icd, dict):
        value = icd.get("ideal_completion", "")
        return content_to_text(value)
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


def main() -> int:
    args = parse_args()

    if os.path.exists(args.output) and not args.overwrite and not args.skip_rebuild:
        print(f"Output already exists: {args.output}. Use --overwrite or --skip-rebuild.", file=sys.stderr)
        return 2

    # Initialize Client & Processor
    endpoint = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
    api_key = os.getenv("AZURE_AI_FOUNDRY_API_KEY")
    if not endpoint or not api_key:
        print("Missing Azure credentials.", file=sys.stderr)
        return 2

    client = build_client(endpoint=endpoint, api_key=api_key, api_version=args.api_version)
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

    # 1. Check existing progress
    existing_ids = set()
    if os.path.exists(args.output):
        print(f"Scanning existing output for resumability...", file=sys.stderr)
        for row in iter_jsonl(args.output):
            if "prompt_id" in row:
                existing_ids.add(row["prompt_id"])

    # 2. Pre-count total rows for the progress bar
    print(f"Counting total records in {args.input}...", file=sys.stderr)
    total_records = sum(1 for _ in open(args.input, 'r', encoding='utf-8'))
    
    total_new_processed = 0
    skipped_records = []
    checkpoint_buffer = []
    checkpoint_interval = 10

    # 3. Process records with tqdm
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

                    # Checkpoint logic
                    if len(checkpoint_buffer) >= checkpoint_interval:
                        write_jsonl(args.output, checkpoint_buffer, mode="a")
                        checkpoint_buffer = []
                    
                    pbar.set_postfix({"new": total_new_processed, "err": len(skipped_records)})
                    pbar.update(1)

                except Exception as e:
                    error_msg = str(e).split('\n')[0]
                    # Use tqdm.write so the log doesn't break the progress bar
                    tqdm.write(f"Skipping {prompt_id} due to error: {error_msg}")
                    skipped_records.append({"id": prompt_id, "error": error_msg})
                    pbar.update(1)
                    continue

    except KeyboardInterrupt:
        print("\n[!] Interrupted. Saving progress...", file=sys.stderr)
    finally:
        # 4. Final Flush
        if checkpoint_buffer:
            write_jsonl(args.output, checkpoint_buffer, mode="a")
            print(f"Final flush: Saved {len(checkpoint_buffer)} rows.", file=sys.stderr)

    print("-" * 30)
    print(f"Run Complete.")
    print(f" - New rows added: {total_new_processed}")
    print(f" - Skipped (errors): {len(skipped_records)}")
    print(f" - Output file: {args.output}")
    print("-" * 30)
    
    return 0

if __name__ == "__main__":
    # Simplified caller since main() now handles its own interrupts safely
    sys.exit(main())
