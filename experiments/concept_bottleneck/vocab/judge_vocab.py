#!/usr/bin/env python
"""LLM judge: grade every term on two scales, using Apertus (or any chat LLM) as annotator.

  valid_p     0..1   probability that the term is a real, standard concept (not an invented,
                     glued, contradictory or fragmentary term). Filter on this FIRST.
  obs_grade   1..5   observability in Sentinel-2 (1 = not observable ... 5 = directly observable)
  fire_grade -2..+2  influence on wildfire in Canada (-2 = strongly reduces ... +2 = strongly increases)

Everything is read from next-token probabilities of the answer digits (no sampling, deterministic):
  *_grade = most probable digit, *_score = expected value, *_conf = probability of that digit.
The grades of an invalid term (low valid_p) are not meaningful: an invalid term tends to land on
obs 1 and fire 0, which would otherwise be confused with 'not observable' and 'neutral'.

Usage:
  python judge_vocab.py --csv vocab_unique.csv --column term \
      --categories s2_canada_vocabulary_categories_extended.csv --category-column primary_category
  python judge_vocab.py --csv vocab_unique.csv --column term --limit 200   # quick test
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "You are a careful expert annotator. Answer with a single digit and nothing else."

VALID_PROMPT = """\
You are an expert in land cover, ecology and remote sensing terminology.

Term: "{term}"{context}

Is this a valid term: a standard, meaningful concept that names a specific kind of land cover, \
vegetation, landform, water body, snow or ice, disturbance or land use, as an ecologist or a \
land-cover legend would? State words such as "dry" or "frozen" and colour + feature phrases such as \
"reddish canopy" are valid.
Answer 0 (invalid) if the term is:
- an invented, misspelled or glued word that is not a standard word, even if you can guess what it means
- too vague or generic to be a concept: it names no specific land feature (e.g. "pink orange object", \
"colorful stuff", "bright area", "interesting feature")
- internally contradictory or meaningless
- a sentence, clause or fragment, or a measurement or index rather than something on the ground
- a poetic or unusual description rather than a recognised concept
Otherwise answer 1 (valid).

Answer with a single digit (0 or 1)."""

OBS_PROMPT = """\
You are an expert in remote sensing. Sentinel-2 provides RGB, red-edge, NIR and SWIR bands at \
10-20 m pixels with a 5-day revisit, so single images and seasonal time series are available.

Term: "{term}"{context}

How observable is this term in Sentinel-2 imagery of Canada?
1 = not observable: too small, hidden (e.g. below the canopy or underground) or not a physical feature
2 = hardly inferable: needs distinctions finer than 10-20 m allows (e.g. a specific plant species, \
individual stems) or relies on weak indirect cues
3 = partly observable: inferable from spectral, seasonal or contextual cues, but ambiguous
4 = clearly observable under good conditions
5 = directly and unambiguously observable with a distinct spectral or spatial signature (e.g. "lake")

Answer with a single digit (1-5)."""

FIRE_PROMPT = """\
You are an expert in wildfire science and Canadian fire ecology (fuel type, fuel moisture, \
ignition sources, fire spread and intensity, natural and artificial fuel breaks).

Term: "{term}"{context}

What is the typical influence of this on wildfire occurrence, spread or intensity in Canada?
1 = strongly reduces fire (e.g. "open water", "snow cover")
2 = somewhat reduces fire (e.g. "green deciduous forest")
3 = no clear influence, or the direction depends too much on context
4 = somewhat increases fire (e.g. "continuous conifer canopy")
5 = strongly increases fire (e.g. "dry cured grass", "dead standing conifers")

Answer with a single digit (1-5)."""


def load(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"  # we read the logits of the LAST position, so pad on the left
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
    model.eval()
    return model, tok


def token_ids(tok, symbols):
    """Token id of each answer symbol (digits); they must be single tokens."""
    ids = [tok.encode(d, add_special_tokens=False) for d in symbols]
    if any(len(i) != 1 for i in ids):
        raise SystemExit(f"The symbols {symbols} are not single tokens for this tokenizer; "
                         "use letters instead.")
    return [i[0] for i in ids]


@torch.inference_mode()
def grade_probs(model, tok, prompts, ids, batch_size, desc):
    """Return (probabilities over the answer symbols, probability mass the model put on any of them)."""
    texts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True) for p in prompts]
    order = np.argsort([len(t) for t in texts])[::-1]  # longest first: an OOM shows up immediately
    probs = np.zeros((len(texts), len(ids)), dtype=np.float32)
    mass = np.zeros(len(texts), dtype=np.float32)
    for s in tqdm(range(0, len(order), batch_size), desc=desc):
        idx = order[s:s + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        try:  # only compute the logits of the last position (saves a lot of memory)
            out = model(**enc, use_cache=False, logits_to_keep=1)
        except TypeError:
            out = model(**enc, use_cache=False)
        logp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)[:, ids]
        mass[idx] = logp.exp().sum(-1).cpu().numpy()
        probs[idx] = torch.softmax(logp, dim=-1).cpu().numpy()
    return probs, mass


def summarize(probs, offset):
    """grade (most probable level), score (expected level), conf (its probability).
    offset 0 -> levels 1..5 ; offset 3 -> levels -2..+2."""
    levels = np.arange(1, probs.shape[1] + 1) - offset
    return levels[probs.argmax(1)], probs @ levels, probs.max(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--column", required=True, help="column with the text to judge")
    ap.add_argument("--out", default=None, help="default: <csv>_judged.csv")
    ap.add_argument("--model", default="swiss-ai/Apertus-8B-Instruct-2509")
    ap.add_argument("--categories", default=None, help="categories CSV (category_id, category, definition)")
    ap.add_argument("--category-column", default=None, help="column of --csv holding a category_id")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N unique items")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    cols = [a.column] + ([a.category_column] if a.category_column else [])
    items = df[cols].dropna(subset=[a.column]).drop_duplicates().reset_index(drop=True)
    if a.limit:
        items = items.head(a.limit)

    ctx = {}
    if a.categories:
        c = pd.read_csv(a.categories)
        ctx = {int(r.category_id): f"{r.category}: {r.definition}" for r in c.itertuples()}

    def context(row):
        try:
            return f"\nCategory it was listed under: {ctx[int(row[a.category_column])]}"
        except (KeyError, TypeError, ValueError):
            return ""

    use_ctx = bool(a.category_column and ctx)
    contexts = [context(r) if use_ctx else "" for _, r in items.iterrows()]
    terms = items[a.column].astype(str).tolist()
    valid_prompts = [VALID_PROMPT.format(term=t, context=c) for t, c in zip(terms, contexts)]
    obs_prompts = [OBS_PROMPT.format(term=t, context=c) for t, c in zip(terms, contexts)]
    fire_prompts = [FIRE_PROMPT.format(term=t, context=c) for t, c in zip(terms, contexts)]

    model, tok = load(a.model)
    ids = token_ids(tok, "12345")

    probs, mass = grade_probs(model, tok, valid_prompts, token_ids(tok, "01"), a.batch_size,
                              "judging validity")
    if mass.mean() < 0.5:
        print(f"WARNING: only {mass.mean():.0%} of the probability is on 0/1 for 'valid'.")
    items["valid_p"] = probs[:, 1].round(3)

    for name, prompts, offset in (("obs", obs_prompts, 0), ("fire", fire_prompts, 3)):
        probs, mass = grade_probs(model, tok, prompts, ids, a.batch_size, f"judging {name}")
        if mass.mean() < 0.5:
            print(f"WARNING: only {mass.mean():.0%} of the probability is on the digits 1-5 for "
                  f"'{name}': the model often does not answer with a digit; check the prompt.")
        items[f"{name}_grade"], items[f"{name}_score"], items[f"{name}_conf"] = summarize(probs, offset)
        items[f"{name}_score"] = items[f"{name}_score"].round(2)
        items[f"{name}_conf"] = items[f"{name}_conf"].round(3)

    out = df.merge(items, on=cols, how="inner" if a.limit else "left")
    out_path = a.out or str(Path(a.csv).with_name(Path(a.csv).stem + "_judged.csv"))
    out.to_csv(out_path, index=False)

    print(f"wrote {len(out)} rows ({len(items)} unique items) to {out_path}")
    print(f"valid_p < 0.5 for {(items['valid_p'] < 0.5).mean():.0%} of unique items")
    ok = out[out["valid_p"] >= 0.5]
    print("grades of the valid terms:")
    print(pd.crosstab(ok["obs_grade"], ok["fire_grade"], margins=True))


if __name__ == "__main__":
    main()