import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

SYSTEM = ("You output only a plain text list, one term per line, "
          "with no numbering, bullets, markdown, or commentary.")

PROMPT = """\
Build a controlled vocabulary of terms that describe what is visible in Sentinel-2 imagery \
(RGB, red edge, NIR, SWIR; 10-20 m pixels) of Canada. The vocabulary is a fixed set of concept \
names for a satellite-image model, so terms must be plain descriptive noun phrases.

Category: {category}
Definition: {definition}
Term length: exactly {n} word(s) per term (a hyphenated word counts as one word)
{examples_block}{exclude_block}
Rules:
- Each term names something an annotator could see or infer at 10-20 m resolution.
- Noun phrases only: no clauses, no verbs, no hedging.
- No proper nouns, no plural/singular duplicates.
- Use ordinary English words only: never invent a word or glue several words together.
- No plant, crop or animal species names: describe vegetation by its form, structure or use.
- Do not merely recombine the example terms: propose other concepts of this category.
- Lowercase, no punctuation other than hyphens.

Write exactly {k} {new}terms, one per line, then stop."""

TERM_RE = re.compile(r"^[a-z][a-z\-]*(?: [a-z][a-z\-]*){0,2}$")


def clean(line: str):
    """Strip bullets/numbering, normalise, and keep only valid 1-3 word lowercase terms."""
    t = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t if TERM_RE.match(t) else None


def example_sets(pool, n_sets, k, rng):
    """n_sets example subsets of size k: consecutive chunks of one shuffled pool (wrapping)."""
    if not pool or k <= 0:
        return [[] for _ in range(n_sets)]
    order = pool[:]
    rng.shuffle(order)
    k = min(k, len(order))
    return [[order[(i * k + j) % len(order)] for j in range(k)] for i in range(n_sets)]


def build_prompt(category, definition, n, k, examples, exclude):
    examples_block = ("\nExample terms from this category (for style only):\n"
                      + "\n".join(examples) + "\n") if examples else ""
    exclude_block = ("\nTerms already collected (do NOT repeat these or close variants; "
                     "give different concepts):\n" + "\n".join(exclude) + "\n") if exclude else ""
    return PROMPT.format(category=category, definition=definition, n=n, k=k,
                         examples_block=examples_block, exclude_block=exclude_block,
                         new="NEW " if exclude else "")


def load(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:  # generate() needs a pad id to fill finished sequences
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
    return model, tok


@torch.inference_mode()
def generate(model, tok, prompt, n_samples, max_new_tokens, temperature, top_p):
    text = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs, do_sample=True, temperature=temperature, top_p=top_p,
        num_return_sequences=n_samples, max_new_tokens=max_new_tokens,
        repetition_penalty=1.0,  # lists legitimately repeat tokens ("forest ...", newline)
        pad_token_id=tok.pad_token_id)
    prompt_len = inputs["input_ids"].shape[1]
    return [tok.decode(seq[prompt_len:], skip_special_tokens=True) for seq in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="swiss-ai/Apertus-8B-Instruct-2509")
    ap.add_argument("--categories", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--samples", type=int, default=4, help="independent sequences per prompt")
    ap.add_argument("--terms-per-call", type=int, default=50)
    ap.add_argument("--example-sets", type=int, default=5, help="different example subsets per n")
    ap.add_argument("--n-examples", type=int, default=6, help="examples per prompt (0 = none)")
    ap.add_argument("--exclusion-rounds", type=int, default=2)
    ap.add_argument("--exclude-size", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cats = pd.read_csv(a.categories)
    model, tok = load(a.model)
    max_new = a.terms_per_call * 10 + 100  # ~10 tokens/term incl. newline, with headroom

    for seed in a.seeds:
        for _, row in tqdm(cats.iterrows(), total=len(cats), desc=f"seed {seed}"):
            cid = int(row["category_id"])
            path = out_dir / f"cat{cid:02d}_seed{seed}.jsonl"
            if path.exists():  # resume support
                continue
            set_seed(seed * 1000 + cid)
            rng = random.Random(seed * 1000 + cid)
            examples_all = [e.strip() for e in str(row["examples"]).split(";") if e.strip()]
            pool = {n: [e for e in examples_all if len(e.split()) == n] for n in (1, 2, 3)}
            sets = {n: example_sets(pool[n], a.example_sets, a.n_examples, rng)
                    for n in (1, 2, 3)}
            seen, recs = set(), []

            def call(n, examples, rnd, set_id, exclude):
                prompt = build_prompt(row["category"], row["definition"], n,
                                      a.terms_per_call, examples, exclude)
                new = 0
                texts = generate(model, tok, prompt, a.samples, max_new, a.temperature, a.top_p)
                for i, txt in enumerate(texts):
                    pos = 0  # rank of the term in the generated list (to check quality drift)
                    for line in txt.splitlines():
                        t = clean(line)
                        if not t:
                            continue
                        recs.append(dict(category_id=cid, term=t, requested_n=n, round=rnd,
                                         example_set=set_id, seed=seed, sample=i, pos=pos))
                        pos += 1
                        if t not in seen:
                            seen.add(t)
                            new += 1
                return new

            new0 = sum(call(n, sets[n][s], 0, s, None)
                       for s in range(a.example_sets) for n in (1, 2, 3))
            tqdm.write(f"cat {cid} seed {seed} round 0: {new0} unique terms")
            for r in range(1, a.exclusion_rounds + 1):
                new_r = 0
                for n in (1, 2, 3):
                    same_n = sorted(t for t in seen if len(t.split()) == n)
                    excl = rng.sample(same_n, min(len(same_n), a.exclude_size))
                    ex = rng.sample(pool[n], min(len(pool[n]), a.n_examples)) if a.n_examples else []
                    new_r += call(n, ex, r, -1, excl)
                tqdm.write(f"cat {cid} seed {seed} round {r}: {new_r} new, {len(seen)} total")

            tmp = path.with_suffix(".tmp")
            tmp.write_text("\n".join(json.dumps(x) for x in recs) + "\n")
            tmp.rename(path)


if __name__ == "__main__":
    main()