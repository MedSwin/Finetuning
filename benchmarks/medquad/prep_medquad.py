#!/usr/bin/env python3
import os
import sys
import json
import pandas as pd

"""
Prep MedQuAD from a single Parquet file into JSONL:

Input:  Parquet with columns:
  document_id, document_source, document_url, category,
  umls_cui, umls_semantic_types, umls_semantic_group,
  synonyms, question_id, question_focus, question_type,
  question, answer

Output JSONL fields (per line):
  id, question, answer,
  url, type, focus, category,
  document_id, document_source,
  umls_cui, umls_semantic_types, umls_semantic_group,
  synonyms

Exe:
python scripts/prep_medquad.py \
  data/medquad/medquad.parquet \
  data/medquad/processed/medquad_clean.jsonl
"""

def norm_or_none(x):
    """Convert NaN/None to None, strip strings; empty -> None."""
    if x is None:
        return None
    # Handle pandas NaN
    try:
        if isinstance(x, float) and pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x).strip()
    return s if s else None

def main():
    if len(sys.argv) != 3:
        print("Usage: prep_medquad.py <input_parquet> <out_jsonl>")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    if not os.path.exists(in_path):
        print(f"ERROR: input parquet not found: {in_path}")
        sys.exit(1)

    print(f"[info] Loading Parquet: {in_path}")
    df = pd.read_parquet(in_path)

    required = ["question_id", "question", "answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: missing required columns in parquet: {missing}")
        sys.exit(1)

    n_total = len(df)
    n_kept = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, row in df.iterrows():
            q = norm_or_none(row.get("question"))
            a = norm_or_none(row.get("answer"))
            if not q or not a:
                # Skip rows without usable Q/A
                continue

            qid  = norm_or_none(row.get("question_id"))
            doc_id = norm_or_none(row.get("document_id"))

            if qid:
                rec_id = qid
            elif doc_id:
                rec_id = f"{doc_id}_{idx}"
            else:
                rec_id = str(idx)

            rec = {
                "id": rec_id,
                "question": q,
                "answer": a,
                # useful metadata (bench script will just ignore these)
                "url": norm_or_none(row.get("document_url")),
                "type": norm_or_none(row.get("question_type")),
                "focus": norm_or_none(row.get("question_focus")),
                "category": norm_or_none(row.get("category")),
                "document_id": doc_id,
                "document_source": norm_or_none(row.get("document_source")),
                "umls_cui": norm_or_none(row.get("umls_cui")),
                "umls_semantic_types": norm_or_none(row.get("umls_semantic_types")),
                "umls_semantic_group": norm_or_none(row.get("umls_semantic_group")),
                "synonyms": norm_or_none(row.get("synonyms")),
            }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_kept += 1

    print(f"Done. Parsed={n_total}, kept(with Q&A)={n_kept}, out={out_path}")

if __name__ == "__main__":
    main()
