import re
import sys
import os
import glob
from typing import List, Optional, Tuple
from collections import Counter, defaultdict

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import spacy
import pke
from huggingface_hub import hf_hub_download, list_repo_files

from msclip.inference.utils import build_model
from src.constants import CONFIG_PATH

# ----------------------------------------------------------------------
# Label utilities: normalization + simple filters
# ----------------------------------------------------------------------

GENERIC_LABELS = {
    "area", "areas", "region", "regions", "zone", "zones",
    "image", "images", "scene", "scenes", "view", "views",
    "satellite image", "satellite view", "satellite imagery",
    "landscape", "landscape scene",
}

COLOR_WORDS = {
    "brown", "blue", "green", "black", "white", "grey", "gray",
    "red", "yellow", "orange",
}

from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List, Optional

import pke
from tqdm import tqdm


# ----------------------------------------------------------------------
# Worker state
# ----------------------------------------------------------------------

_EXTRACTOR = None
_EXTRACTOR_TYPE = None
_MAX_NGRAM = None
_TOP_K = None


def normalize_label_str(s: str) -> str:
    s = s.strip().lower()
    s = " ".join(s.split())
    for art in ("the ", "a ", "an "):
        if s.startswith(art):
            s = s[len(art):]
    return s


def is_bad_label(s: str) -> bool:
    s = normalize_label_str(s)
    if not s:
        return True
    if s in GENERIC_LABELS:
        return True
    if s in COLOR_WORDS:
        return True
    if len(s) <= 2:
        return True
    return False

# ------------------------------------------------------------------------------
# Download sentences (parquet captions) from the SSL4EO dataset on HuggingFace
# ------------------------------------------------------------------------------

def download_captions(output_dir: str):
    """Download the main parquet files"""

    dataset_repo = "ibm-esa-geospatial/Llama3-SSL4EO-S12-v1.1-captions"
    filename_pattern = re.compile(r".*\.parquet")

    # List files in dataset repo
    all_files = list_repo_files(repo_id=dataset_repo, repo_type="dataset")
    print(f"Found {len(all_files)} files in the dataset repository.")
    tot_files = [f for f in all_files if filename_pattern.search(os.path.basename(f))]
    print(f"Filtered down to {len(tot_files)} parquet files matching pattern.")

    os.makedirs(output_dir, exist_ok=True)

    local_files = []
    for file in tqdm(tot_files, total=len(tot_files)):
        local_path = hf_hub_download(repo_id=dataset_repo, filename=file, repo_type="dataset", local_dir=output_dir)
        local_files.append(local_path)
        print(f"Downloaded: {local_path}")

    return local_files


# ----------------------------------------------------------------------
# Load sentences (parquet captions) from the SSL4EO dataset
# ----------------------------------------------------------------------

def load_sentences_from_parquet(
    captions_dir: str,
    pattern: str = "*.parquet",
    max_phrases: Optional[int] = None,
) -> List[str]:
    paths = sorted(glob.glob(os.path.join(captions_dir, pattern)))
    if not paths:
        raise RuntimeError(f"No parquet files found under {captions_dir} / {pattern}")

    dfs = [pd.read_parquet(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    phrases: List[str] = []
    taken = False
    for col in ["caption", "captions", "text", "description", "prompt"]:
        if col in df.columns:
            phrases.extend(df[col].dropna().astype(str).tolist())
            taken = True

    if not taken:
        for col in df.columns:
            if df[col].dtype == object:
                phrases.extend(df[col].dropna().astype(str).tolist())

    seen = set()
    uniq: List[str] = []
    for s in phrases:
        s = s.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        uniq.append(s)

    if max_phrases is not None and len(uniq) > max_phrases:
        uniq = uniq[:max_phrases]

    print(f"[INFO] Loaded {len(uniq)} unique sentences.")
    return uniq


# ----------------------------------------------------------------------
# MS-CLIP sentence vs candidate phrases (KeyBert like)
# ----------------------------------------------------------------------

@torch.no_grad()
def highlight_phrases_with_msclip(
    sentence: str,
    candidates: List[str],
    tokenizer: nn.Module,
    model: nn.Module,
    top_k: int = 1,
    ngram: int = 5,
) -> List[str]:
    """
    Given one sentence and a list of candidate phrases:
      - encode the sentence with MS-CLIP
      - encode all candidates
      - return the top_k candidates with highest cosine similarity.
    """
    if not candidates:
        return []

    # encode sentence
    toks_sent = tokenizer([sentence]).to(model.device)
    sent_emb = model.inference_text(toks_sent)
    sent_emb = F.normalize(sent_emb, dim=-1)[0]  # [D]

    # encode candidates
    toks = tokenizer(candidates).to(model.device)
    cand_embs = model.inference_text(toks)
    cand_embs = F.normalize(cand_embs, dim=-1)   # [C, D]

    sims = cand_embs @ sent_emb                  # [C]

    counts = torch.tensor([len(s.split()) for s in candidates], device=sims.device)

    out = []
    for n in range(1, ngram + 1):
        idx = (counts == n).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue   # <-- skip, don't append []

        k_n = min(top_k, idx.numel())
        top_local = torch.topk(sims[idx], k=k_n).indices
        top_global = idx[top_local].cpu().tolist()
        out.extend([candidates[j] for j in top_global])


    return out


def build_term_vocab_spacy(
    sentences: List[str],
    model: Optional[nn.Module] = None,
    tokenizer: Optional[nn.Module] = None,
    min_freq: int = 5,
    max_terms: Optional[int] = 100_000,
    max_ngram: int = 3,
    use_msclip: bool = False,
    top_k_per_sentence: int = 1,
    n_process: int = 1,
    msclip_batch_size: int = 512,
    use_parser: bool = False,
) -> Tuple[List[str], List[int]]:

    print(f"[INFO] Loading spaCy model (n_process={n_process})...")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    # ------------------------------------------------------------------
    # Pass 1: spaCy extraction — collect per-sentence candidates
    # ------------------------------------------------------------------
    print("[INFO] Extracting candidates with spaCy...")
    per_sentence: List[tuple] = []  # (sentence_str, [candidate_phrases])

    chunk_count = 0
    filter_count = 0
    lemmas_count = 0
    is_bad_count = 0
    tot_chunk = []
    tot_crop_chunk = []

    for sent, doc in tqdm(
        zip(sentences, nlp.pipe(sentences, batch_size=512, n_process=n_process)),
        total=len(sentences),
    ):
        if use_parser:
            ### New implementation: use noun chunks, filter + lemmatize, no n-gram limit but max span length = max_ngram | Drop redundant filters ###
            candidate_phrases: List[str] = []
            for chunk in doc.noun_chunks:
                tot_chunk.append(chunk)
                chunk_count += 1
                if len(chunk) > max_ngram:
                    chunk = chunk[len(chunk)-max_ngram:]
                    tot_crop_chunk.append(chunk)

                filtered = [t for t in chunk if t.is_alpha and not t.is_stop and t.pos_ in {"NOUN", "ADJ"}]

                if not filtered:
                    filter_count += 1
                    continue

                # Lemma deduplication within span
                lemmas = [t.lemma_.lower() for t in filtered]
                if len(set(lemmas)) != len(lemmas):
                    lemmas_count += 1
                    continue

                phrase = normalize_label_str(" ".join(t.lemma_ for t in filtered))
                if not is_bad_label(phrase):
                    candidate_phrases.append(phrase)
                else:
                    is_bad_count += 1

        else:
            ### Original Louis' implementation: n-grams of content words (NOUN/ADJ), no parsing ###
            content_toks = [
                tok for tok in doc
                if tok.is_alpha
                and not tok.is_stop
                and tok.pos_ in {"NOUN", "ADJ"}
            ]

            candidate_phrases: List[str] = []
            for n in range(1, max_ngram + 1):
                if len(content_toks) < n:
                    continue
                for i in range(len(content_toks) - n + 1):
                    span = content_toks[i:i + n]
                    head = span[-1]

                    lemmas = [t.lemma_.lower() for t in span]
                    if len(set(lemmas)) != len(lemmas):
                        continue
                    if head.pos_ not in {"NOUN"}:
                        continue

                    phrase = normalize_label_str(" ".join(t.lemma_ for t in span))
                    if is_bad_label(phrase):
                        continue
                    if not any(t.pos_ in {"NOUN"} for t in span):
                        continue

                    candidate_phrases.append(phrase)

        if candidate_phrases:
            per_sentence.append((sent, candidate_phrases))

    counts: Counter = Counter()
    print(f"Total Chunks: {chunk_count}, Filtered: {filter_count}, Lemma Duplicates: {lemmas_count}, Bad Labels: {is_bad_count}")
    print(f"[INFO] Extracted candidates for {len(per_sentence)} sentences.")
    print(f"[INFO] Number of total candidates across all sentences: {sum(len(cands) for _, cands in per_sentence)}")
    print(f"[INFO] Number of unique candidate phrases across all sentences: {len(set(p for _, cands in per_sentence for p in cands))}")
    print(f"[INFO] Total chunks: {len(tot_chunk)}")
    print(f"[INFO] Number of unique chunks: {len(set(t for t in tot_chunk))}")
    print(f"[INFO] Cropped chunks: {len(tot_crop_chunk)}")
    print(f"[INFO] Number of unique cropped chunks: {len(set(t for t in tot_crop_chunk))}")
    #raise Exception("Debug stop - remove after checking counts")

    if not use_msclip:
        # ------------------------------------------------------------------
        # No MS-CLIP: just count all candidates directly
        # ------------------------------------------------------------------
        for _, candidates in per_sentence:
            for p in candidates:
                counts[p] += 1

    else:
        # ------------------------------------------------------------------
        # Pass 2: batch MS-CLIP encoding
        # ------------------------------------------------------------------
        assert model is not None and tokenizer is not None

        # 2a. Collect all unique phrases and sentences
        all_phrases = sorted({p for _, cands in per_sentence for p in cands})
        all_sents   = [s for s, _ in per_sentence]

        phrase2idx = {p: i for i, p in enumerate(all_phrases)}

        print(f"[INFO] Encoding {len(all_phrases)} unique phrases and "
              f"{len(all_sents)} sentences with MS-CLIP...")

        @torch.no_grad()
        def batch_encode_text(texts: List[str], batch_size: int) -> torch.Tensor:
            embs = []
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
                batch = texts[i:i + batch_size]
                toks = tokenizer(batch).to(model.device)
                e = model.inference_text(toks)
                e = F.normalize(e, dim=-1)
                embs.append(e.cpu())
            return torch.cat(embs, dim=0)  # [N, D]

        phrase_embs = batch_encode_text(all_phrases, msclip_batch_size)  # [P, D]
        sent_embs   = batch_encode_text(all_sents,   msclip_batch_size)  # [S, D]

        # Precompute word counts per phrase for ngram grouping
        phrase_lengths = torch.tensor([len(p.split()) for p in all_phrases])  # [P]

        # 2b. Per sentence: lookup precomputed embeddings, find top-k
        print("[INFO] Computing similarities and selecting top-k...")
        for s_idx, (sent, candidates) in enumerate(tqdm(per_sentence)):
            sent_emb = sent_embs[s_idx]                          # [D]

            # Get indices into all_phrases for this sentence's candidates
            cand_ids = torch.tensor([phrase2idx[p] for p in candidates])  # [C]

            cand_embs   = phrase_embs[cand_ids]                  # [C, D]
            sims        = cand_embs @ sent_emb                   # [C]
            cand_lengths = phrase_lengths[cand_ids]              # [C]

            for n in range(1, max_ngram + 1):
                mask = (cand_lengths == n).nonzero(as_tuple=True)[0]
                if mask.numel() == 0:
                    continue
                k_n = min(top_k_per_sentence, mask.numel())
                top_local  = torch.topk(sims[mask], k=k_n).indices
                top_global = mask[top_local].tolist()
                for j in top_global:
                    counts[candidates[j]] += 1


    freqs = np.array(list(counts.values()))
    print(f"Total unique phrases before min_freq: {len(freqs)}")
    print(f"Phrases appearing >= 5 times:   {(freqs >= 5).sum()}")
    print(f"Phrases appearing >= 50 times:  {(freqs >= 50).sum()}")
    print(f"Phrases appearing >= 500 times: {(freqs >= 500).sum()}")
    print(f"Median frequency: {np.median(freqs):.0f}")
    print(f"Mean frequency:   {np.mean(freqs):.0f}")
    print(f"Max frequency:    {np.max(freqs):.0f}")

    items_sorted = sorted(counts.items(), key=lambda x: -x[1])
    n = len(items_sorted)

    print("Top 20 most frequent:")
    for phrase, count in items_sorted[:20]:
        print(f"  {count:6d}  {phrase}")

    print("\n20 from middle of distribution:")
    for phrase, count in items_sorted[n//2 - 10: n//2 + 10]:
        print(f"  {count:6d}  {phrase}")

    print("\n20 least frequent (just above min_freq):")
    for phrase, count in items_sorted[-20:]:
        print(f"  {count:6d}  {phrase}")

    # ------------------------------------------------------------------
    # Frequency filtering + sorting
    # ------------------------------------------------------------------
    items = [(p, c) for p, c in counts.items() if c >= min_freq]
    items.sort(key=lambda x: -x[1])

    if max_terms is not None and len(items) > max_terms:
        print("CAREFULLLLLLLL Cropped list")
        items = items[:max_terms]

    terms = [p for p, _ in items]
    freqs = [f for _, f in items]
    print(f"[INFO] Built spaCy term vocabulary of size {len(terms)} "
          f"(min_freq={min_freq}).")
    return terms, freqs


def _init_worker(
    extractor_type: str,
    max_ngram: int,
    top_k_per_sentence: int,
):
    global _EXTRACTOR
    global _EXTRACTOR_TYPE
    global _MAX_NGRAM
    global _TOP_K

    _EXTRACTOR_TYPE = extractor_type
    _MAX_NGRAM = max_ngram
    _TOP_K = top_k_per_sentence

    if extractor_type == "TopicRank":
        _EXTRACTOR = pke.unsupervised.TopicRank()
    elif extractor_type == "MultipartiteRank":
        _EXTRACTOR = pke.unsupervised.MultipartiteRank()
    elif extractor_type == "Yake":
        _EXTRACTOR = pke.unsupervised.YAKE()
    else:
        raise ValueError(f"Unsupported extractor type: {extractor_type}")


def _process_sentence(sent: str):
    global _EXTRACTOR
    global _EXTRACTOR_TYPE
    global _MAX_NGRAM
    global _TOP_K

    try:
        _EXTRACTOR.load_document(
            input=sent,
            language="en",
        )

        if _EXTRACTOR_TYPE in {"TopicRank", "MultipartiteRank"}:
            _EXTRACTOR.candidate_selection()
        else:
            _EXTRACTOR.candidate_selection(n=_MAX_NGRAM)

        _EXTRACTOR.candidate_weighting()

        keyphrases = _EXTRACTOR.get_n_best(n=_TOP_K)

        if not keyphrases:
            return None

        candidate_phrases = []

        for phrase, _score in keyphrases:
            words = phrase.split()

            if len(words) > _MAX_NGRAM:
                phrase = " ".join(words[-_MAX_NGRAM:])

            if is_bad_label(phrase):
                continue

            candidate_phrases.append(
                normalize_label_str(phrase)
            )

        if not candidate_phrases:
            return None

        return candidate_phrases

    except Exception:
        # Optional:
        # logging.exception(...)
        return None


def keyphrase_extraction(
    extractor_type: str,
    sentences: List[str],
    min_freq: int = 5,
    max_terms: Optional[int] = 100_000,
    max_ngram: int = 3,
    top_k_per_sentence: int = 1,
    num_workers: Optional[int] = None,
    chunksize: int = 10,
) -> Tuple[List[str], List[int]]:
    """
    Multiprocessing implementation using imap_unordered.
    """

    if num_workers is None:
        num_workers = cpu_count()

    print(
        f"[INFO] Loading Extractor={extractor_type}, "
        f"workers={num_workers}, chunksize={chunksize}"
    )

    counts = Counter()

    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(
            extractor_type,
            max_ngram,
            top_k_per_sentence,
        ),
    ) as pool:

        iterator = pool.imap_unordered(
            _process_sentence,
            sentences,
            chunksize=chunksize,
        )

        for candidate_phrases in tqdm(
            iterator,
            total=len(sentences),
        ):
            if candidate_phrases:
                counts.update(candidate_phrases)

    items = [
        (phrase, count)
        for phrase, count in counts.items()
        if count >= min_freq
    ]

    items.sort(key=lambda x: -x[1])

    if max_terms is not None and len(items) > max_terms:
        print(
            f"[WARN] Cropping vocabulary "
            f"from {len(items)} to {max_terms}"
        )
        items = items[:max_terms]

    terms = [phrase for phrase, _ in items]
    freqs = [freq for _, freq in items]

    print(
        f"[INFO] Built PKE term vocabulary "
        f"of size {len(terms)} "
        f"(min_freq={min_freq})"
    )

    return terms, freqs


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="extract_concept")
def extract_concepts(cfg: DictConfig):

    if cfg.download_captions:
        print("[INFO] Downloading captions from HuggingFace...")
        download_captions(cfg.captions_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    captions_dir = cfg.captions_dir
    dict_dir = cfg.dict_dir
    sentences = load_sentences_from_parquet(captions_dir)

    if cfg.use_msclip:
        msclip_model, _, tokenizer = build_model(
                model_name=cfg.model.model_name, pretrained=cfg.model.pretrained,
                ckpt_path=cfg.model.ckpt_path, device=device, channels=cfg.model.channels
        )
        msclip_model.to(device).eval()
    else:
        msclip_model, tokenizer = None, None


    if cfg.method_type == "spacy":

        # --------- Build term vocabulary BEFORE k-means ----------
        terms, freqs = build_term_vocab_spacy(
            sentences,
            model=msclip_model,
            tokenizer=tokenizer,
            min_freq=cfg.min_freq,
            max_terms=cfg.max_terms,
            max_ngram=cfg.max_ngram,
            use_msclip=cfg.use_msclip,
            top_k_per_sentence=cfg.top_k_per_sentence,
            n_process=os.cpu_count(),
            msclip_batch_size=cfg.msclip_batch_size,
            use_parser=cfg.use_parser
        )
        out_stub = "spacy" if not cfg.use_msclip else "spacy_msclip"

    elif cfg.method_type == "pke":
        terms, freqs = keyphrase_extraction(
            extractor_type=cfg.pke_extractor,
            sentences=sentences,
            min_freq=cfg.min_freq,
            max_terms=cfg.max_terms,
            max_ngram=cfg.max_ngram,
            top_k_per_sentence=cfg.top_k_per_sentence,
        )
        out_stub = f"pke_{cfg.pke_extractor.lower()}"

    else:
        raise NotImplementedError()


    # --------- Save dictionary ----------
    os.makedirs(dict_dir, exist_ok=True)
    out_path = os.path.join(dict_dir, f"{out_stub}_general_f{cfg.min_freq}k{cfg.top_k_per_sentence}n{cfg.max_ngram}max{cfg.max_terms}parser{cfg.use_parser}_homogeneous.csv")

    df = pd.DataFrame({
        "concept": terms,
        "frequency": freqs
    })

    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved {len(terms)} concepts to {out_path}")

if __name__ == "__main__":
    extract_concepts()