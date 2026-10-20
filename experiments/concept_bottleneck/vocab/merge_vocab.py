import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:  # optional: pip install wordfreq
    from wordfreq import zipf_frequency
except ImportError:
    zipf_frequency = None

try:  # optional: pip install spacy && python -m spacy download en_core_web_sm
    import spacy
except ImportError:
    spacy = None

FUNCTION_WORDS = {"with", "in", "within", "of", "like", "and", "or", "for", "on", "by", "to",
                  "from", "the", "a", "an"}
POS_TEMPLATE = "This image shows a {}."


def singular(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "sses", "xes")):
        return w[:-2]
    if w.endswith("s") and len(w) > 3 and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def key(term: str) -> str:
    """Duplicate key = sorted letters of the singularised, glued words. Merges word-order,
    hyphen, space and glued variants ('forest clearcut' == 'clearcut-forest' == 'clearcutforest').
    Terms that are anagrams of each other collide too (rare: ~0.1% here, mostly synonyms such as
    'brushland'/'shrubland'); the `variants` column lets you audit them."""
    return "".join(sorted("".join(singular(w) for w in term.replace("-", " ").split())))


def words(term: str) -> set:
    return {singular(w) for w in term.replace("-", " ").split()}


def quality_flags(term: str, seed_words: set):
    """(unknown_word, clause): invented/glued words (not in English and not a seed word),
    and function words or 2+ hyphens (clauses, hyphen chains). Flags only, nothing is dropped."""
    ws = term.replace("-", " ").split()
    if zipf_frequency is not None:
        unknown = any(zipf_frequency(w, "en") == 0 and singular(w) not in seed_words for w in ws)
    else:  # crude fallback without wordfreq
        unknown = any(len(w) >= 13 for w in ws)
    return unknown, (any(w in FUNCTION_WORDS for w in ws) or term.count("-") >= 2)


def best_form(forms, seed_words):
    """Representative spelling: a variant made of real words first, then the most frequent,
    then the one with more spaces."""
    return max(Counter(forms).items(),
               key=lambda kv: (not quality_flags(kv[0], seed_words)[0], kv[1], kv[0].count(" ")))[0]


def not_noun_flags(nlp, terms, allowed):
    """Multiword terms only: True if the last word is tagged as something other than a noun.
    A sentence template stabilises the tagging of noun/verb homographs ('stand', 'patch').
    Skipped (False): 1-grams, a hyphenated last word, and last words that are seed-example
    words anywhere ('plain', 'flat', 'outcrop'...). Out-of-context tagging is noisy: flag only."""
    check = [len(t.split()) > 1 and "-" not in t.split()[-1] and singular(t.split()[-1]) not in allowed
             for t in terms]
    todo = [t for t, c in zip(terms, check) if c]
    docs = nlp.pipe([POS_TEMPLATE.format(t) for t in todo], batch_size=512)
    tags = {t: d[-2].pos_ for t, d in zip(todo, docs)}  # d[-1] is the final period
    return [c and tags[t] not in ("NOUN", "PROPN") for t, c in zip(terms, check)]


def flag_near_duplicates(entries, model, thr):
    """Flag only: near_dup_of = the closest MORE supported term if cosine >= thr. Nothing is
    merged and no seeds are added: embedders rate antonyms ('dry soil' / 'moist soil',
    'evergreen' / 'deciduous') as very close, so a human or a judge should decide."""
    for e in entries:
        e["near_dup_of"], e["near_dup_sim"] = "", np.nan
    order = sorted(range(len(entries)), key=lambda i: (-len(entries[i]["seeds"]),
                                                       -len(entries[i]["runs"]),
                                                       -entries[i]["mentions"]))
    emb = model.encode([entries[i]["term"] for i in order], normalize_embeddings=True)
    sims = emb @ emb.T
    for rank in range(1, len(order)):
        j = int(sims[rank, :rank].argmax())
        if sims[rank, j] >= thr:
            e = entries[order[rank]]
            e["near_dup_of"] = entries[order[j]]["term"]
            e["near_dup_sim"] = round(float(sims[rank, j]), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--categories", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-runs", type=int, default=0,
                    help="0 keeps terms found only via exclusion rounds; 1+ drops them")
    ap.add_argument("--min-seeds", type=int, default=0, help="filter out terms with fewer seeds")
    ap.add_argument("--embed-model", default=None, help="optional: flag near-duplicates (no merge)")
    ap.add_argument("--sim", type=float, default=0.93)
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--no-pos", action="store_true", help="skip the not_noun flag")
    ap.add_argument("--n-perm", type=int, default=20)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for p in sorted(Path(a.raw).glob("cat*_seed*.jsonl"))
            for l in p.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(recs)
    df["run"] = df["seed"].astype(str) + "-" + df["sample"].astype(str)
    df["key"] = df["term"].map(key)
    if "round" not in df:
        df["round"] = 0

    cats = pd.read_csv(a.categories)
    seed_terms = {int(r.category_id): [e.strip() for e in str(r.examples).split(";") if e.strip()]
                  for r in cats.itertuples()}
    seed_keys = {c: {key(t) for t in ts} for c, ts in seed_terms.items()}
    seed_words = {c: set().union(*[words(t) for t in ts]) for c, ts in seed_terms.items()}
    all_seed_words = set().union(*seed_words.values())

    embedder = None
    if a.embed_model:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(a.embed_model)

    nlp = None
    if not a.no_pos:
        if spacy is None:
            print("spaCy not installed: skipping the not_noun flag")
        else:
            try:
                nlp = spacy.load(a.spacy_model, disable=["parser", "ner", "lemmatizer"])
            except OSError:
                print(f"spaCy model '{a.spacy_model}' not found: skipping the not_noun flag")

    rows, sat_rows = [], []
    for cid, g in df.groupby("category_id"):
        sw = seed_words.get(cid, set())
        # Support is computed on round 0 only: exclusion rounds deliberately avoid seen terms,
        # so counting them would bias support downward for exactly the terms they target.
        n_runs_total = g.loc[g["round"] == 0, "run"].nunique()
        entries = []
        for k, gk in g.groupby("key"):
            forms = gk["term"].tolist()
            entries.append(dict(key=k, term=best_form(forms, sw), mentions=len(gk),
                                variants=set(forms),
                                runs=set(gk.loc[gk["round"] == 0, "run"]),
                                seeds=set(gk["seed"])))

        for e in entries:
            e["is_seed"] = e["key"] in seed_keys.get(cid, set())
            e["n_novel_words"] = len(words(e["term"]) - sw)
            e["unknown_word"], e["clause"] = quality_flags(e["term"], sw)
        if nlp is not None:
            for e, f in zip(entries, not_noun_flags(nlp, [e["term"] for e in entries],
                                                    all_seed_words)):
                e["not_noun"] = f
        if embedder is not None:
            flag_near_duplicates(entries, embedder, a.sim)

        non_seed = [e for e in entries if not e["is_seed"]]
        recomb = sum(e["n_novel_words"] == 0 for e in non_seed) / max(len(non_seed), 1)

        # Good-Turing: share of observations that are singletons ~ P(next draw is new)
        n1 = sum(len(e["runs"]) == 1 for e in entries)
        N = sum(len(e["runs"]) for e in entries)
        n_excl = sum(len(e["runs"]) == 0 for e in entries)
        print(f"category {cid}: {len(entries)} unique terms ({n_excl} only via exclusion rounds) "
              f"from {n_runs_total} round-0 runs | P(new term next run) ~ {n1 / max(N, 1):.2f} | "
              f"seed-word recombinations: {recomb:.0%} | "
              f"unknown-word terms: {np.mean([e['unknown_word'] for e in entries]):.0%} | "
              f"not-noun: {np.mean([e.get('not_noun', False) for e in entries]):.1%} | "
              f"near-dups: {sum(bool(e.get('near_dup_of')) for e in entries)}")

        # Saturation curve: cumulative unique terms as runs are added (random orders)
        per_run = defaultdict(set)
        for e in entries:
            for r in e["runs"]:
                per_run[r].add(e["key"])
        runs = list(per_run)
        if runs:
            curves = []
            for _ in range(a.n_perm):
                random.shuffle(runs)
                seen, c = set(), []
                for r in runs:
                    seen |= per_run[r]
                    c.append(len(seen))
                curves.append(c)
            for i, v in enumerate(np.mean(curves, axis=0), start=1):
                sat_rows.append(dict(category_id=cid, n_runs=i, mean_unique_terms=round(v, 1)))

        for e in entries:
            rows.append(dict(
                category_id=cid, term=e["term"], n_gram=len(e["term"].split()),
                n_seeds=len(e["seeds"]), n_runs=len(e["runs"]), n_runs_total=n_runs_total,
                support=round(len(e["runs"]) / max(n_runs_total, 1), 3),
                n_mentions=e["mentions"], exclusion_only=len(e["runs"]) == 0,
                is_seed=e["is_seed"], n_novel_words=e["n_novel_words"],
                seed_recombination=(e["n_novel_words"] == 0 and not e["is_seed"]),
                unknown_word=e["unknown_word"], clause=e["clause"],
                suspect=bool(e["unknown_word"] or e["clause"]),
                not_noun=e.get("not_noun", False),
                near_dup_of=e.get("near_dup_of", ""), near_dup_sim=e.get("near_dup_sim", np.nan),
                variants="; ".join(sorted(e["variants"] - {e["term"]}))))

    vocab = pd.DataFrame(rows)
    vocab["n_categories"] = vocab.groupby(vocab["term"].map(key))["category_id"].transform("nunique")
    vocab = vocab[vocab["n_seeds"] >= a.min_seeds]
    vocab = vocab[vocab["n_runs"] >= a.min_runs]
    sort_cols = ["category_id", "n_seeds", "n_runs", "n_mentions"]
    vocab = vocab.sort_values(sort_cols, ascending=[True, False, False, False])
    vocab.to_csv(out / "vocab_merged.csv", index=False)
    pd.DataFrame(sat_rows).to_csv(out / "saturation.csv", index=False)

    # One row per term across categories: primary category = the seed-example category if any,
    # else the category with the most seeds / runs / mentions. `categories` lists all of them.
    gk_ = vocab["term"].map(key)
    best = (vocab.assign(gkey=gk_)
            .sort_values(["gkey", "is_seed", "n_seeds", "n_runs", "n_mentions"],
                         ascending=[True, False, False, False, False])
            .drop_duplicates("gkey"))
    cats_of = vocab.assign(gkey=gk_).groupby("gkey")["category_id"].agg(
        lambda s: "; ".join(map(str, sorted(set(s)))))
    unique = (best.assign(categories=best["gkey"].map(cats_of))
              .rename(columns={"category_id": "primary_category"}).drop(columns="gkey")
              .sort_values(["primary_category", "n_seeds", "n_runs", "n_mentions"],
                           ascending=[True, False, False, False]))
    unique.to_csv(out / "vocab_unique.csv", index=False)
    print(f"wrote {len(vocab)} (category, term) rows to {out / 'vocab_merged.csv'} "
          f"and {len(unique)} unique terms to {out / 'vocab_unique.csv'}")


if __name__ == "__main__":
    main()