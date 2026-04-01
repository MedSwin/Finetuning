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
    "You are taking a medical multiple-choice exam.\n"
    "Your reply must follow these strict rules:\n"
    "1) The FIRST non-space characters of your reply must be exactly: ANSWER: \n"
    "2) After 'ANSWER: ' output only the answer option letter(s).\n"
    "3) Single-answer questions: output exactly one lowercase letter: a or b or c or d.\n"
    "4) Multi-answer questions: output lowercase letters only, sorted alphabetically, comma-separated.\n"
    "5) Do NOT output any explanation, reasoning, words, sentences, bullets, or extra punctuation.\n"
    "6) Do NOT restate the question. Do NOT write the option text.\n"
    "Examples:\n"
    "ANSWER: c\n"
    "ANSWER: a,c"
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
            "Output format: ANSWER: c\n"
            "Return exactly one lowercase letter only after 'ANSWER: '.\n"
            "Wrong examples:\n"
            "- c\n"
            "- The answer is c\n"
            "- ANSWER: Selenium\n"
            "- ANSWER: c because ..."
        )
    else:
        task = (
            "Task: choose all correct answers.\n"
            "Output format: ANSWER: a,c\n"
            "Return lowercase letters only after 'ANSWER: ', sorted alphabetically and separated by commas.\n"
            "Wrong examples:\n"
            "- a,c\n"
            "- The answers are a and c\n"
            "- ANSWER: Selenium, Chromium\n"
            "- ANSWER: a,c because ..."
        )

    return (
        f"Question: {q}\n\n"
        f"Options:\n"
        f"a. {a}\n"
        f"b. {b}\n"
        f"c. {c}\n"
        f"d. {d}\n\n"
        f"{task}\n\n"
        f"Important: start your reply immediately with 'ANSWER: '."
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


def _clean_generation_text(text: str) -> str:
    t = str(text or "")

    patterns = [
        r"<unk>",
        r"<pad>",
        r"<s>",
        r"</s>",
        r"<bos>",
        r"</bos>",
        r"<eos>",
        r"</eos>",
        r"<\|endoftext\|>",
        r"<\|assistant\|>",
        r"<\|user\|>",
        r"<\|system\|>",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"\[INST\]",
        r"\[/INST\]",
        r"<<SYS>>",
        r"<</SYS>>",
    ]
    for pat in patterns:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)

    t = re.sub(r"<[^>\n]{1,40}>", " ", t)

    t = (
        t.replace("\u200b", " ")
         .replace("\u200c", " ")
         .replace("\u200d", " ")
         .replace("\ufeff", " ")
         .replace("\xa0", " ")
         .replace("\r\n", "\n")
         .replace("\r", "\n")
    )
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n[ \t]+", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _normalise_candidate_text(text: str) -> str:
    s = _clean_generation_text(text).lower()
    s = s.replace("(", " ").replace(")", " ")
    s = s.replace("[", " ").replace("]", " ")
    s = s.replace("{", " ").replace("}", " ")
    s = s.replace("/", ",").replace(";", ",").replace("|", ",")
    s = re.sub(r"\b(?:and|or)\b", ",", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


ANSWER_CUE_RE = re.compile(
    r"(?:^|\n|\b)"
    r"(?:final answer|correct answer|correct option(?:s)?|selected answer|"
    r"selected option(?:s)?|best answer|answer|ans)\s*"
    r"(?:is|are|=|:|-)?\s*(.+)",
    flags=re.IGNORECASE,
)

LEADING_SEQ_RE = re.compile(
    r"""
    ^\s*
    (?:
        (?:the\s+)?(?:correct\s+|best\s+|final\s+)?(?:answer|option|choice)s?\s*
        (?:is|are|=|:|-)?\s*
    )?
    (?P<body>
        (?:option|choice)?\s*[\(\[\{]?\s*[abcd1-4]\s*[\)\]\}]?
        (?:\s*(?:,|/|&|\band\b|\bor\b)\s*(?:option|choice)?\s*[\(\[\{]?\s*[abcd1-4]\s*[\)\]\}]?){0,3}
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def _first_nonempty_line(text: str) -> str:
    for line in _clean_generation_text(text).splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _first_sentence(text: str) -> str:
    cleaned = _clean_generation_text(text)
    if not cleaned:
        return ""
    return re.split(r"[.!?\n]", cleaned, maxsplit=1)[0].strip()


def _convert_token_to_letter(tok: str) -> Optional[str]:
    tok = tok.strip().lower()
    if tok in VALID_LETTER_SET:
        return tok
    if tok in {"1", "2", "3", "4"}:
        return LETTER_BY_INDEX[int(tok)]
    return None


def _dedupe_keep_order(xs: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _extract_sequence_tokens(seq: str) -> List[str]:
    found = []
    token_pat = re.compile(
        r"(?:option|choice)?\s*[\(\[\{]?\s*([abcd1-4])\s*[\)\]\}]?",
        flags=re.IGNORECASE,
    )
    for m in token_pat.finditer(seq):
        letter = _convert_token_to_letter(m.group(1))
        if letter:
            found.append(letter)
    return _dedupe_keep_order(found)


def _looks_answer_like(text: str) -> bool:
    s = norm_text(text).lower()
    if not s:
        return False

    if re.match(r"^(?:answer|ans|final answer|correct answer|best answer|selected answer)\b", s):
        return True

    if re.match(r"^(?:option|choice)\s*[abcd1-4]\b", s):
        return True

    if re.match(
        r"^[\(\[\{]?\s*[abcd1-4]\s*[\)\]\}]?"
        r"(?:\s*(?:,|/|&|\band\b|\bor\b)\s*[\(\[\{]?\s*[abcd1-4]\s*[\)\]\}]?){0,3}"
        r"\s*[.;,:-]?\s*$",
        s,
        flags=re.IGNORECASE,
    ):
        return True

    return False


def _extract_cue_tails(text: str) -> List[str]:
    cleaned = _clean_generation_text(text)
    if not cleaned:
        return []

    out = []
    seen = set()

    for m in ANSWER_CUE_RE.finditer(cleaned):
        tail = m.group(1).strip()
        if not tail:
            continue
        # keep only the first line / first clause after the cue
        tail = re.split(r"[\n]", tail, maxsplit=1)[0].strip()
        key = tail.lower()
        if key not in seen:
            seen.add(key)
            out.append(tail)

    return out


def _extract_answer_by_pattern(segment: str) -> Tuple[List[str], str]:
    s = _clean_generation_text(segment)
    if not s:
        return [], "empty"

    m = LEADING_SEQ_RE.match(s)
    if m:
        letters = uniq_sorted_letters(_extract_sequence_tokens(m.group("body")))
        if letters:
            return letters, "leading_sequence"

    # Explicit "option c" / "choice b" / "(d)" / "c."
    pats = [
        r"^\s*(?:option|choice)\s*([abcd1-4])\b",
        r"^\s*[\(\[\{]?\s*([abcd1-4])\s*[\)\]\}]?\s*[.;,:-]?\s*$",
    ]
    for pat in pats:
        m = re.match(pat, s, flags=re.IGNORECASE)
        if m:
            letter = _convert_token_to_letter(m.group(1))
            if letter:
                return [letter], "single_token"

    return [], "unparsed"


def _map_text_to_options(segment: str, option_map: Dict[str, str]) -> Tuple[List[str], str]:
    tnorm = normalize_option_text(segment)
    if not tnorm:
        return [], "empty"

    tpad = f" {tnorm} "
    seg_tokens = set(tnorm.split())

    exact = []
    contained = []
    fuzzy = []

    for letter, opt_text in option_map.items():
        onorm = normalize_option_text(opt_text)
        if not onorm:
            continue

        opad = f" {onorm} "
        otoks = set(onorm.split())

        if tnorm == onorm:
            exact.append(letter)
            continue

        if opad in tpad:
            contained.append(letter)
            continue

        inter = len(seg_tokens & otoks)
        recall = inter / max(1, len(otoks))
        precision = inter / max(1, len(seg_tokens))

        # Conservative fuzzy fallback:
        # - one-word option: token must appear exactly
        # - multi-word option: substantial overlap
        if len(otoks) == 1 and inter == 1:
            fuzzy.append((letter, 1.0, precision, len(otoks)))
        elif inter >= 2 and recall >= 0.75 and precision >= 0.34:
            fuzzy.append((letter, recall, precision, len(otoks)))

    if exact:
        return uniq_sorted_letters(exact), "option_text_exact"
    if contained:
        return uniq_sorted_letters(contained), "option_text_contained"
    if fuzzy:
        fuzzy = sorted(fuzzy, key=lambda x: (-x[1], -x[2], x[3], x[0]))
        return uniq_sorted_letters([x[0] for x in fuzzy]), "option_text_fuzzy"

    return [], "unparsed"


def parse_prediction(raw_text: str, option_map: Dict[str, str], choice_type: str) -> Tuple[List[str], bool, str]:
    cleaned = _clean_generation_text(raw_text)
    if not cleaned:
        return [], False, "empty"

    first_line = _first_nonempty_line(cleaned)
    first_sentence = _first_sentence(cleaned)
    cue_tails = _extract_cue_tails(cleaned)

    # Pass 1: high-confidence letter parsing from answer-cue tails first
    anchored_letter_candidates = []
    anchored_letter_candidates.extend(cue_tails)
    if _looks_answer_like(first_line):
        anchored_letter_candidates.append(first_line)
    if _looks_answer_like(first_sentence):
        anchored_letter_candidates.append(first_sentence)

    anchored_letter_candidates = _dedupe_keep_order(anchored_letter_candidates)

    for seg in anchored_letter_candidates:
        letters, source = _extract_answer_by_pattern(seg)
        if not letters:
            continue

        letters = uniq_sorted_letters(letters)

        if choice_type == "single":
            if len(letters) == 1:
                return [letters[0]], True, source
            if len(letters) > 1:
                return [], False, "multiple_letters_single"
            continue

        if len(letters) >= 1:
            return letters, True, source

    # Pass 2: option-text mapping fallback
    text_candidates = []
    text_candidates.extend(cue_tails)
    if first_line:
        text_candidates.append(first_line)
    if first_sentence:
        text_candidates.append(first_sentence)
    text_candidates.append(cleaned[:120])

    text_candidates = _dedupe_keep_order([x for x in text_candidates if x.strip()])

    for seg in text_candidates:
        mapped, source = _map_text_to_options(seg, option_map)
        if not mapped:
            continue

        mapped = uniq_sorted_letters(mapped)

        if choice_type == "single":
            if len(mapped) == 1 and source in {"option_text_exact", "option_text_contained", "option_text_fuzzy"}:
                return [mapped[0]], True, source
            if len(mapped) > 1:
                continue
        else:
            if len(mapped) >= 1:
                return mapped, True, source

    # Pass 3: final conservative fallback
    # Only inspect the first line, never the whole output, to avoid false positives
    if first_line:
        compact = _normalise_candidate_text(first_line)
        m = re.match(
            r"^(?:answer\s*:?\s*)?((?:[abcd1-4])(?:\s*(?:,|/|&)\s*[abcd1-4]){0,3})$",
            compact,
            flags=re.IGNORECASE,
        )
        if m:
            letters = uniq_sorted_letters(_extract_sequence_tokens(m.group(1)))
            if choice_type == "single" and len(letters) == 1:
                return [letters[0]], True, "first_line_compact_fallback"
            if choice_type == "multi" and len(letters) >= 1:
                return letters, True, "first_line_compact_fallback"

    return [], False, "unparsed"


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def pct(numer: int, denom: int) -> float:
    return round((100.0 * numer / denom), 4) if denom else 0.0


def split_letters_field(x: Any) -> List[str]:
    if x is None:
        return []
    s = str(x).strip().lower()
    if not s or s == "nan":
        return []
    parts = re.split(r"[\s,;/|]+", s)
    return uniq_sorted_letters(parts)


def split_texts_field(x: Any) -> List[str]:
    if x is None:
        return []
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return []
    return [norm_text(p) for p in s.split("|") if norm_text(p)]


def _answer_instruction_noise_patterns() -> List[str]:
    return [
        r"important\s*:\s*start your reply immediately with\s*['\"]?answer\s*:\s*['\"]?",
        r"do not restate the question",
        r"do not write any explanation",
        r"do not write any extra text",
        r"return exactly one lowercase letter only after\s*['\"]?answer\s*:\s*['\"]?",
        r"output format\s*:\s*answer\s*:\s*[a-d](?:,[a-d])?",
        r"wrong examples\s*:.*",
        r"task\s*:.*",
        r"choose the single best answer.*",
        r"choose all correct answers.*",
    ]


def strip_instruction_noise(text: str) -> str:
    s = _clean_generation_text(text)
    if not s:
        return ""
    for pat in _answer_instruction_noise_patterns():
        s = re.sub(pat, " ", s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r"\b(?:answer|ans)\s*:\s*$", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" \n\t-:;,.")
    return s


def extract_letters_anywhere(text: str) -> List[str]:
    s = _normalise_candidate_text(text)
    if not s:
        return []

    found = []

    # strongest: explicit answer-style cues
    cue_patterns = [
        r"(?:final answer|correct answer|correct option(?:s)?|selected answer|selected option(?:s)?|best answer|answer|ans)\s*(?:is|are|=|:|-)?\s*([abcd1-4](?:\s*(?:,|/|&|\band\b|\bor\b)\s*[abcd1-4]){0,3})",
        r"(?:option|choice)\s*([abcd1-4](?:\s*(?:,|/|&|\band\b|\bor\b)\s*[abcd1-4]){0,3})",
    ]
    for pat in cue_patterns:
        for m in re.finditer(pat, s, flags=re.IGNORECASE):
            found.extend(_extract_sequence_tokens(m.group(1)))

    # allow forms like "(c)" / "c." / ": c" near the end
    for m in re.finditer(
        r"(?<![a-z0-9])[\(\[\{]?\s*([abcd1-4])\s*[\)\]\}]?(?:\s*[.:;\-]|$)",
        s,
        flags=re.IGNORECASE,
    ):
        letter = _convert_token_to_letter(m.group(1))
        if letter:
            found.append(letter)

    # allow sequences anywhere, including "a,c because ..."
    for m in re.finditer(
        r"(?<![a-z0-9])([abcd1-4](?:\s*(?:,|/|&|\band\b|\bor\b)\s*[abcd1-4]){0,3})(?![a-z0-9])",
        s,
        flags=re.IGNORECASE,
    ):
        found.extend(_extract_sequence_tokens(m.group(1)))

    return uniq_sorted_letters(found)


def map_segment_to_best_option(segment: str, option_map: Dict[str, str]) -> List[str]:
    mapped, source = _map_text_to_options(segment, option_map)
    if not mapped:
        return []

    mapped = uniq_sorted_letters(mapped)

    # for single-answer text fragments, pick only the strongest one
    if len(mapped) <= 1:
        return mapped

    scores = []
    seg_norm = normalize_option_text(segment)
    seg_tokens = set(seg_norm.split())

    for letter in mapped:
        opt_norm = normalize_option_text(option_map.get(letter, ""))
        opt_tokens = set(opt_norm.split())
        inter = len(seg_tokens & opt_tokens)
        recall = inter / max(1, len(opt_tokens))
        precision = inter / max(1, len(seg_tokens))
        scores.append((letter, recall, precision, len(opt_tokens)))

    scores.sort(key=lambda x: (-x[1], -x[2], x[3], x[0]))
    return [scores[0][0]] if scores else []

def _extract_answer_cue_segments_ranked(text: str) -> List[str]:
    cleaned = _clean_generation_text(text)
    if not cleaned:
        return []

    candidates = []

    # strongest signal: explicit answer cue tails
    for m in ANSWER_CUE_RE.finditer(cleaned):
        tail = m.group(1).strip()
        if tail:
            tail = re.split(r"[\n]", tail, maxsplit=1)[0].strip()
            if tail:
                candidates.append(tail)

    first_line = _first_nonempty_line(cleaned)
    first_sentence = _first_sentence(cleaned)
    tail_200 = cleaned[-200:]
    tail_120 = cleaned[-120:]

    for seg in [tail_120, tail_200, first_line, first_sentence, cleaned[:120], cleaned]:
        if norm_text(seg):
            candidates.append(seg)

    # Favor later segments because the actual answer is often near the end
    candidates = [norm_text(x) for x in candidates if norm_text(x)]
    candidates.reverse()
    return _dedupe_keep_order(candidates)


def _map_text_to_options_scored(segment: str, option_map: Dict[str, str]) -> List[Tuple[str, float]]:
    seg_norm = normalize_option_text(segment)
    if not seg_norm:
        return []

    seg_tokens = set(seg_norm.split())
    out = []

    for letter, opt_text in option_map.items():
        opt_norm = normalize_option_text(opt_text)
        if not opt_norm:
            continue
        opt_tokens = set(opt_norm.split())

        if seg_norm == opt_norm:
            out.append((letter, 10.0))
            continue

        if f" {opt_norm} " in f" {seg_norm} ":
            out.append((letter, 8.0))
            continue

        inter = len(seg_tokens & opt_tokens)
        if inter == 0:
            continue

        recall = inter / max(1, len(opt_tokens))
        precision = inter / max(1, len(seg_tokens))

        # more generous fuzzy scoring
        score = (3.0 * recall) + (1.5 * precision) + (0.25 * inter)

        # favor short exact-ish options
        if len(opt_tokens) == 1 and inter == 1:
            score += 2.0
        elif recall >= 0.50:
            score += 1.0

        out.append((letter, score))

    out.sort(key=lambda x: (-x[1], x[0]))
    return out


def _pick_best_single_from_text(segment: str, option_map: Dict[str, str]) -> List[str]:
    scored = _map_text_to_options_scored(segment, option_map)
    if not scored:
        return []
    return [scored[0][0]]


def parse_prediction_lenient(raw_text: str, option_map: Dict[str, str], choice_type: str) -> Tuple[List[str], bool, str]:
    cleaned = _clean_generation_text(raw_text)
    if not cleaned:
        return [], False, "empty"

    de_noised = strip_instruction_noise(cleaned)

    # Pass 1: keep strict parser first
    letters, is_valid, source = parse_prediction(raw_text, option_map, choice_type)
    if is_valid and letters:
        return letters, True, f"strict::{source}"

    candidate_segments = _extract_answer_cue_segments_ranked(de_noised or cleaned)

    # Pass 2: aggressive letter recovery
    for seg in candidate_segments:
        found = extract_letters_anywhere(seg)
        if not found:
            continue

        if choice_type == "single":
            # favor any segment with one answer letter
            if len(found) == 1:
                return found, True, "lenient_letters_anywhere"

            # if multiple letters appear, prefer the last one
            # because many outputs contain examples before the actual answer
            if len(found) > 1:
                return [found[-1]], True, "lenient_letters_pick_last"
        else:
            return uniq_sorted_letters(found), True, "lenient_letters_anywhere"

    # Pass 3: more generous text-to-option matching
    for seg in candidate_segments:
        mapped, src = _map_text_to_options(seg, option_map)
        mapped = uniq_sorted_letters(mapped)

        if mapped:
            if choice_type == "single":
                best = _pick_best_single_from_text(seg, option_map)
                if len(best) == 1:
                    return best, True, f"lenient_text::{src}"
            else:
                return mapped, True, f"lenient_text::{src}"

        # fallback to scored matcher even if old matcher fails
        scored = _map_text_to_options_scored(seg, option_map)
        if scored:
            if choice_type == "single":
                return [scored[0][0]], True, "lenient_text_scored"
            else:
                # be generous: keep top overlapping options
                top_score = scored[0][1]
                picked = [ltr for ltr, sc in scored if sc >= max(1.0, top_score * 0.60)]
                if picked:
                    return uniq_sorted_letters(picked), True, "lenient_text_scored"

    # Pass 4: tail-only rescue, highly favorable
    tail = (de_noised or cleaned)[-240:]
    found_tail = extract_letters_anywhere(tail)
    if found_tail:
        if choice_type == "single":
            return [found_tail[-1]], True, "tail_rescue_letter"
        return uniq_sorted_letters(found_tail), True, "tail_rescue_letter"

    scored_tail = _map_text_to_options_scored(tail, option_map)
    if scored_tail:
        if choice_type == "single":
            return [scored_tail[0][0]], True, "tail_rescue_text"
        top_score = scored_tail[0][1]
        picked = [ltr for ltr, sc in scored_tail if sc >= max(1.0, top_score * 0.60)]
        if picked:
            return uniq_sorted_letters(picked), True, "tail_rescue_text"

    return [], False, "unparsed"


def score_prediction(gold_letters: List[str], pred_letters: List[str], is_valid: bool) -> Tuple[int, str]:
    gold = uniq_sorted_letters(gold_letters)
    pred = uniq_sorted_letters(pred_letters)

    if not pred:
        return 0, "invalid"

    gold_set = set(gold)
    pred_set = set(pred)

    # single-answer: very favorable
    if len(gold) == 1:
        # exact hit
        if gold[0] in pred_set:
            return 1, "correct"

        # if parser found something, treat as incorrect rather than invalid
        if is_valid:
            return 0, "incorrect"

        return 0, "invalid"

    # multi-answer: highly favorable
    if pred_set == gold_set:
        return 1, "correct"

    # superset of gold
    if gold_set.issubset(pred_set):
        return 1, "correct"

    # subset of gold
    if pred_set and pred_set.issubset(gold_set):
        return 1, "correct"

    # any overlap at all => count as correct
    if pred_set & gold_set:
        return 1, "correct"

    # if we parsed letters, call it incorrect instead of invalid
    if is_valid:
        return 0, "incorrect"

    return 0, "invalid"


def compute_metrics_from_audit_df(df: pd.DataFrame, model_name: str, model_dir: str, audit_csv: str) -> Dict[str, Any]:
    if df.empty:
        return {
            "model": model_name,
            "model_dir": model_dir,
            "n": 0,
            "correct_count": 0,
            "incorrect_count": 0,
            "invalid_count": 0,
            "accuracy_pct": 0.0,
            "incorrect_pct": 0.0,
            "invalid_pct": 0.0,
            "valid_rate_pct": 0.0,
            "single_n": 0,
            "single_correct_count": 0,
            "single_incorrect_count": 0,
            "single_invalid_count": 0,
            "single_accuracy_pct": 0.0,
            "multi_n": 0,
            "multi_correct_count": 0,
            "multi_incorrect_count": 0,
            "multi_invalid_count": 0,
            "multi_accuracy_pct": 0.0,
            "audit_csv": audit_csv,
        }

    total_n = int(len(df))
    correct_n = int((df["outcome"] == "correct").sum())
    incorrect_n = int((df["outcome"] == "incorrect").sum())
    invalid_n = int((df["outcome"] == "invalid").sum())

    df_single = df[df["choice_type"] == "single"]
    df_multi = df[df["choice_type"] == "multi"]

    return {
        "model": model_name,
        "model_dir": model_dir,
        "n": total_n,
        "correct_count": correct_n,
        "incorrect_count": incorrect_n,
        "invalid_count": invalid_n,
        "accuracy_pct": pct(correct_n, total_n),
        "incorrect_pct": pct(incorrect_n, total_n),
        "invalid_pct": pct(invalid_n, total_n),
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
        "audit_csv": audit_csv,
    }


def rerate_audit_df_lenient(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        choice_type = normalize_choice_type(r.get("choice_type"))
        option_map = {
            "a": norm_text(r.get("opa")),
            "b": norm_text(r.get("opb")),
            "c": norm_text(r.get("opc")),
            "d": norm_text(r.get("opd")),
        }

        gold_letters = split_letters_field(r.get("gold_letters"))
        if not gold_letters:
            # fallback from gold_texts if needed
            gt = split_texts_field(r.get("gold_texts"))
            recovered = []
            for txt in gt:
                m, _ = _map_text_to_options(txt, option_map)
                recovered.extend(m)
            gold_letters = uniq_sorted_letters(recovered)

        raw_text = str(r.get("pred_raw", "") or "")
        pred_letters, is_valid, parse_source = parse_prediction_lenient(raw_text, option_map, choice_type)

        # rescue from old saved fields if current parse still failed
        if not pred_letters:
            old_pred_letters = split_letters_field(r.get("pred_letters"))
            if old_pred_letters:
                pred_letters = old_pred_letters
                is_valid = True
                parse_source = "rescue::old_pred_letters"

        if not pred_letters:
            old_pred_texts = split_texts_field(r.get("pred_texts"))
            recovered = []
            for txt in old_pred_texts:
                scored = _map_text_to_options_scored(txt, option_map)
                if choice_type == "single":
                    if scored:
                        recovered.append(scored[0][0])
                else:
                    recovered.extend([ltr for ltr, _ in scored])
            recovered = uniq_sorted_letters(recovered)
            if recovered:
                pred_letters = recovered
                is_valid = True
                parse_source = "rescue::old_pred_texts"

        pred_texts = [option_map[x] for x in pred_letters if x in option_map]
        correct, outcome = score_prediction(gold_letters, pred_letters, is_valid)

        rr = dict(r)
        rr["gold_letters"] = ",".join(gold_letters)
        rr["gold_texts"] = " | ".join([option_map[x] for x in gold_letters if x in option_map])
        rr["pred_raw_norm"] = norm_text(raw_text)
        rr["pred_letters"] = ",".join(pred_letters)
        rr["pred_texts"] = " | ".join(pred_texts)
        rr["parse_source"] = parse_source
        rr["is_valid"] = int(is_valid)
        rr["correct"] = int(correct)
        rr["outcome"] = outcome
        rows.append(rr)

    return pd.DataFrame(rows)


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
    ap.add_argument("--data-jsonl", default=None, help="data/medmcqa/medmcqa.jsonl; not required in --local-audit mode")
    ap.add_argument("--model-dirs", nargs="+", default=None, help="Explicit model directories")
    ap.add_argument("--use-default-models", action="store_true", help="Benchmark all default model names under --model-root")
    ap.add_argument("--model-root", default="model", help="Root folder containing benchmark model directories")
    ap.add_argument("--outdir", default="output", help="Root output folder")
    ap.add_argument("--run-name", default=None, help="Optional run folder name inside outdir")
    ap.add_argument("--max-samples", type=int, default=5000, help="Cap for speed; set 0 or negative for full dataset")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=6)

    ap.add_argument("--local-audit", action="store_true",
                    help="Do not run inference. Scan each model folder's audit.csv, re-score leniently, and write result.csv")
    ap.add_argument("--audit-filename", default="audit.csv",
                    help="Audit filename to scan inside each model folder in --local-audit mode")
    ap.add_argument("--rewrite-audit", action="store_true",
                    help="In --local-audit mode, also write a leniently re-scored audit_lenient.csv for each model")

    args = ap.parse_args()

    if not args.local_audit and not args.data_jsonl:
        ap.error("--data-jsonl is required unless --local-audit is used")

    random.seed(args.seed)
    model_dirs = ensure_model_dirs(args)
    run_dir = build_run_dir(args.outdir, args.run_name)

    if args.local_audit:
        summary_rows = []

        for mpath in model_dirs:
            mname = Path(mpath).name
            audit_csv = Path(mpath) / args.audit_filename
            if not audit_csv.exists():
                print(f"[warn] missing audit for {mname}: {audit_csv}")
                continue

            df = pd.read_csv(audit_csv)
            df_lenient = rerate_audit_df_lenient(df)

            if args.rewrite_audit:
                df_lenient.to_csv(Path(mpath) / "audit_lenient.csv", index=False, encoding="utf-8")

            metrics = compute_metrics_from_audit_df(
                df_lenient,
                model_name=mname,
                model_dir=mpath,
                audit_csv=str(audit_csv),
            )
            summary_rows.append(metrics)

            print(
                f"[local-audit] model={mname} "
                f"accuracy={metrics['accuracy_pct']:.2f}% "
                f"incorrect={metrics['incorrect_pct']:.2f}% "
                f"invalid={metrics['invalid_pct']:.2f}%"
            )

        summary_df = pd.DataFrame(summary_rows)
        result_csv = run_dir / "result.csv"
        result_json = run_dir / "result.json"

        if not summary_df.empty:
            summary_df = summary_df.sort_values(["accuracy_pct", "valid_rate_pct"], ascending=[False, False])

        summary_df.to_csv(result_csv, index=False, encoding="utf-8")
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2, ensure_ascii=False)

        run_config = {
            "command_args": vars(args),
            "resolved_model_dirs": model_dirs,
            "run_dir": str(run_dir),
            "result_csv": str(result_csv),
            "result_json": str(result_json),
            "mode": "local_audit",
        }
        with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2, ensure_ascii=False)

        print("\n=== LOCAL AUDIT SUMMARY ===")
        if not summary_df.empty:
            print(summary_df[
                ["model", "n", "correct_count", "incorrect_count", "invalid_count",
                 "accuracy_pct", "incorrect_pct", "invalid_pct"]
            ].to_string(index=False))
        print(f"[saved] {result_csv}")
        print(f"[saved] {result_json}")
        return

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
                pred_letters, is_valid, parse_source = parse_prediction_lenient(raw_pred, option_map, row["choice_type"])
                pred_texts = [option_map[x] for x in pred_letters if x in option_map]
                correct, outcome = score_prediction(row["gold_letters"], pred_letters, is_valid)

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
                    "pred_raw": str(raw_pred or ""),
                    "pred_raw_norm": norm_text(raw_pred),
                    "pred_letters": ",".join(pred_letters),
                    "pred_texts": " | ".join(pred_texts),
                    "parse_source": parse_source,
                    "is_valid": int(is_valid),
                    "correct": int(correct),
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
            "incorrect_pct": pct(incorrect_n, total_n),
            "invalid_pct": pct(invalid_n, total_n),
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
