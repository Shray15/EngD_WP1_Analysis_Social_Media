#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
from pathlib import Path
import pandas as pd
import spacy

# ----------------------
# Config (edit paths/cols)
# ----------------------


INPUT_DATA    = r"C:\Users\20245179\OneDrive - TU Eindhoven\Research Paper\final_data_with_author_names.csv"
OUTPUT_CSV   = r"C:\Users\20245179\OneDrive - TU Eindhoven\Research Paper\final_data_cleaned_with_author_names.csv"

TEXT_COL    = "text"        # the text column to clean
ORG_COL     = "author_name"   # where org names come from

ORG_TOKEN   = "<ORG>"
PERSON_TOKEN= "<PERSON>"

# ----------------------
# Load data
# ----------------------
data = pd.read_csv(INPUT_DATA)
full_df = pd.read_csv(INPUT_DATA)  # for org names

if TEXT_COL not in data.columns:
    raise KeyError(f"Column '{TEXT_COL}' not found. Available: {list(data.columns)}")

data[TEXT_COL] = data[TEXT_COL].astype(str).fillna("")

# ----------------------
# Build an org-name regex
# ----------------------
org_names = (
    full_df.get(ORG_COL, pd.Series([], dtype=str))
            .dropna()
            .astype(str)
            .str.strip()
)
# Deduplicate, drop empty, sort longest-first to avoid partial overlaps
org_names = sorted({n for n in org_names if n}, key=len, reverse=True)

if org_names:
    escaped = [re.escape(n) for n in org_names]
    # Whole-word match on letters/digits; avoids matching ING inside THING, but matches "ING-Bank"
    pattern = r"(?<![A-Za-z0-9])(?:%s)(?![A-Za-z0-9])" % "|".join(escaped)
    ORG_RE = re.compile(pattern, flags=re.IGNORECASE)
else:
    ORG_RE = None

def replace_orgs(text: str) -> str:
    if not ORG_RE:
        return text
    return ORG_RE.sub(ORG_TOKEN, text)

# ----------------------
# spaCy NER for person names
# ----------------------
# Install once (outside Python):  python -m spacy download nl_core_news_lg
nlp = spacy.load("nl_core_news_lg", disable=["tagger","lemmatizer","morphologizer","attribute_ruler"])
nlp.enable_pipe("ner")
PERSON_LABELS = {"PER", "PERSON"}

def replace_persons(text_list):
    """Batch replace person entities with <PERSON>."""
    out = []
    for doc in nlp.pipe(text_list, batch_size=128):
        s = doc.text
        # Replace from right to left to keep indices valid
        persons = [ent for ent in doc.ents if ent.label_ in PERSON_LABELS]
        for ent in sorted(persons, key=lambda e: e.start_char, reverse=True):
            s = s[:ent.start_char] + PERSON_TOKEN + s[ent.end_char:]
        out.append(s)
    return out

# ----------------------
# Run the pipeline
# ----------------------
# 1) Replace org names from list
data[TEXT_COL] = data[TEXT_COL].map(replace_orgs)

# 2) Replace person names via NER (batched)
data[TEXT_COL] = replace_persons(data[TEXT_COL].tolist())

# ----------------------
# Save
# ----------------------
Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
data.to_csv(OUTPUT_CSV, index=False)
print(f"Saved cleaned data -> {OUTPUT_CSV}")

# In[ ]:



