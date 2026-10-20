### Switch to Apertus specific èython environment

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import os
from typing import Dict, Union


def load_model_and_tokenizer(model_name: str,
                             precision: torch.dtype = torch.bfloat16,
                             attn_implementation: str = "sdpa"):
    """
    Load the model and tokenizer from the given model name.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=precision,
                                                 attn_implementation=attn_implementation,
                                                 device_map="auto")
    return model, tokenizer


@torch.inference_mode()
def generate_batch(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt_text: str,
                   max_new_tokens: int = 30000,
                   temperature: float = 0.8, top_p: float = 0.9, repetition_penalty: float = 1.,
                   seed: int = None):

    if seed is not None:
        torch.manual_seed(seed)

    messages = [
        {"role": "system", "content": "You output only a plain text list as instructed, one term per line, with no JSON, no markdown formatting, no bullets, no numbering, and no prose outside the specified NOTE lines."},
        {"role": "user", "content": prompt_text},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    return tokenizer.decode(
        output_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )


def build_category_prompt(category, definition, examples):
    return f"""
ROLE
You are assisting in constructing a controlled concept vocabulary,
which will serve as the fixed candidate set of "atoms" a wildfire forecasting deep learning
model can name its internal concepts with. The vocabulary itself is the deliverable
— not prose, not captions.

TASK
Generate 1-to-3-gram English terms (single words, bigrams, or trigrams) that
are visually or physically observable, or strongly inferable, in multispectral
Sentinel-2 satellite imagery (bands: visible RGB, red edge, NIR, SWIR) over
CANADA specifically for the category: "{category}", with the following definition:
"{definition}".

CRITICAL RULES
- Each term names something an annotator could see or infer at 10-20 m resolution.
- Noun phrases only: no clauses, no verbs, no hedging.
- No proper nouns, no species names, no plural/singular duplicates.
- Lowercase, no punctuation other than hyphens.
- Write at least 200 terms, up to ~300 terms.

EXAMPLES
The terms below are examples of 1-to-3-gram terms of the type of output expected,
but they are not exhaustive or prescriptive.

Terms are semicolon-separated below; OUTPUT FORMAT marks the end of the
reference:
{examples}

OUTPUT FORMAT
Output the terms as plain text, one term per line, with no bullets, no
numbering, no markdown formatting, and no surrounding punctuation or quotes.
Write at least 200 terms, up to ~300 terms. Do not include any prose outside."""


def generate_vocabulary(model_name: str, category_csv_path: os.PathLike,
                        output_path: os.PathLike, model_kwargs: Dict[str, Union[str, torch.dtype]] = {},
                        gen_kwargs: Dict[str, Union[int, float]] = {}):
    """
    Generate a controlled concept vocabulary for a given category using a language model.
    """
    model, tokenizer = load_model_and_tokenizer(model_name, **model_kwargs)

    category_df = pd.read_csv(category_csv_path)

    Path(output_path).mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(category_df.iterrows(), total=category_df.shape[0], desc="Generating vocabularies"):
        category_id = row["category_id"]
        category = row["category"]
        definition = row["definition"]
        examples = row["examples"]

        prompt_text = build_category_prompt(category, definition, examples)
        generated_text = generate_batch(model, tokenizer, prompt_text, **gen_kwargs)

        output_file_path = Path(output_path) / f"category_{category_id}_vocab.txt"
        with open(output_file_path, "w") as f:
            f.write(generated_text)

if __name__ == "__main__":
    model_name = "swiss-ai/Apertus-8B-Instruct-2509"
    category_csv_path = "/home/hporta/code_project/wildfire-forecast/CanadaFireSat-Model/data/s2_canada_vocabulary_categories_simple.csv"
    output_path = "/home/hporta/code_project/wildfire-forecast/CanadaFireSat-Model/results/DictionnaryWordsGeneral/apertus_output"

    model_kwargs = {
        "precision": torch.bfloat16,
        "attn_implementation": "sdpa",
    }

    gen_kwargs = {
        "max_new_tokens": 30000,
        "temperature": 0.8,
        "top_p": 0.9,
        "repetition_penalty": 1.,
        "seed": 42,
    }

    generate_vocabulary(model_name, category_csv_path, output_path, model_kwargs, gen_kwargs)
