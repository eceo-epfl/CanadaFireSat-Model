from collections import Counter
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span


BAD_ENT_TYPES = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "FAC",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "NORP",
}

GENERIC_TERMS = {
    "area",
    "areas",
    "region",
    "regions",
    "zone",
    "zones",
    "place",
    "places",
    "location",
    "locations",
    "image",
    "images",
    "picture",
    "pictures",
    "scene",
    "scenes",
    "view",
    "views",
    "satellite image",
    "satellite view",
    "satellite imagery",
}

COLOR_WORDS = {
    "brown", "blue", "green", "black", "white", "grey", "gray",
    "red", "yellow", "orange",
}

CAPTION_VERBS = {
    "show", "shows", "showing",
    "include", "includes", "including",
    "feature", "features", "featuring",
    "contain", "contains", "containing",
    "indicate", "indicates", "indicating",
    "display", "displays", "displaying",
    "depict", "depicts", "depicting",
    "represent", "represents", "representing",
    "surround", "surrounds", "surrounding",
    "appear", "appears", "appearing",
}


def normalize_label_str(s: str) -> str:
    s = s.strip().lower()
    s = " ".join(s.split())
    for art in ("the ", "a ", "an "):
        if s.startswith(art):
            s = s[len(art):]
    return s


def is_english_word(word: str) -> bool:
    word = word.lower().strip()

    if not word.isalpha():
        return False

    return bool(wn.synsets(word))


def postprocess_terms(
    terms,
    freqs,
    nlp,
):
    results = []

    if freqs is None:
        freqs = [1] * len(terms)

    #for phrase, freq in tqdm(zip(terms, freqs), total=len(terms), desc="[INFO] Postprocessing terms"):
    for phrase, freq in zip(terms, freqs):

        if isinstance(phrase, Doc):
            # Already processed by spaCy
            doc = phrase

        elif isinstance(phrase, Span):
            # Already processed by spaCy
            doc = phrase

        elif isinstance(phrase, str):
            # Still need normalization + spaCy processing
            phrase = normalize_label_str(phrase)
            if not phrase:
                continue
            doc = nlp(phrase)

        else:
            raise ValueError(f"Unexpected type for phrase: {type(phrase)}")


        if not doc:
            continue

        # --------------------------------------------------
        # 3. Hard Reject: Proper nouns / named entities
        # --------------------------------------------------
        if any(tok.pos_ == "PROPN" for tok in doc):
            continue

        if any(ent.label_ in BAD_ENT_TYPES for ent in doc.ents):
            continue


        # --------------------------------------------------
        # Dropping: grammatical words / Caption-generation artifacts / Generic singleton concepts
        # --------------------------------------------------

        kept = []
        for tok in doc:

            if not tok.is_alpha:
                continue

            if tok.lemma_.lower() in GENERIC_TERMS:
                continue

            if tok.lemma_.lower() in CAPTION_VERBS:
                continue

            if tok.pos_ in {
                "PRON", "DET", "SCONJ", "INTJ", "AUX"
            }:
                continue

            if not is_english_word(tok.lemma_.lower()):
                continue

            kept.append(tok)


        doc = kept

        # --------------------------------------------------
        # 5. Must be noun-centered
        # --------------------------------------------------
        nouns = [
            tok for tok in doc
            if tok.pos_ == "NOUN"
        ]

        if not nouns:
            continue

        # --------------------------------------------------
        # 8. Pure color concepts
        # --------------------------------------------------
        if len(doc) == 1:
            if doc[0].lemma_.lower() in COLOR_WORDS:
                continue

        # --------------------------------------------------
        # 9. Normalize to lemmas
        # --------------------------------------------------
        concept = " ".join(
            tok.lemma_.lower()
            for tok in doc
        )

        results.append((concept, freq))

    # merge duplicates after lemmatization
    merged = Counter()

    for concept, freq in results:
        merged[concept] += freq

    return sorted(
        merged.items(),
        key=lambda x: -x[1]
    )
