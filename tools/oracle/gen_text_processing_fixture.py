#!/usr/bin/env python3
"""Generate the text_processing oracle fixture (catboost-8z4.43 / P3.6):
deterministic Tokenizer/Dictionary inputs + the pinned Python catboost==1.2.10
catboost.text_processing.Tokenizer/Dictionary observable outputs, covering
the default configuration (Word-level, unigram, FrequencyBased, ByDelimiter
separator) plus number-processing policies, delimiter modes, unknown-token
policy, end-of-sentence policy, max_dictionary_size, save/load round-trip
and fitting without a tokenizer / from pre-tokenized data.

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_text_processing_fixture.py
"""
import json
import os
import sys

import catboost
from catboost.text_processing import Dictionary, Tokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "text_processing.json")
SAVED_DICT_PATH = os.path.join(FIXTURE_DIR, "text_processing_dictionary.freq")

TEXTS = [
    "Cats are beautiful animals.",
    "Dogs are loyal, dogs are great.",
    "The Cat sat on the mat, and the cat purred.",
]

NUMBER_STRING = "I have 12 cats and 3 dogs in 2024"


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    # --- Tokenizer: default (ByDelimiter, delimiter=" ") ---
    tok_default = Tokenizer()
    tokenize_default = [tok_default.tokenize(t) for t in TEXTS]

    # --- Tokenizer: lowercasing ---
    tok_lower = Tokenizer(lowercasing=True, separator_type="ByDelimiter", delimiter=" ")
    tokenize_lower = [tok_lower.tokenize(t) for t in TEXTS]

    # --- Tokenizer: number_process_policy ---
    tokenize_number_skip = Tokenizer(number_process_policy="Skip").tokenize(NUMBER_STRING)
    tokenize_number_replace = Tokenizer(number_process_policy="Replace").tokenize(NUMBER_STRING)
    tokenize_number_replace_custom = Tokenizer(
        number_process_policy="Replace", number_token="<NUM>"
    ).tokenize(NUMBER_STRING)

    # --- Tokenizer: split_by_set ---
    tokenize_split_by_set = Tokenizer(
        separator_type="ByDelimiter", delimiter=",;", split_by_set=True
    ).tokenize("a,b;c,,d")

    # --- Tokenizer: skip_empty = False ---
    tokenize_no_skip_empty = Tokenizer(delimiter=" ", skip_empty=False).tokenize("a  b   c")

    # --- Tokenizer: multi-char delimiter ---
    tokenize_multi_delimiter = Tokenizer(delimiter="::").tokenize("a::b::c")

    # --- Dictionary: default fit/apply/introspection ---
    dict_default = Dictionary(occurence_lower_bound=0)
    dict_default.fit(TEXTS, tok_lower)
    dictionary_default = {
        "size": int(dict_default.size),
        "top_tokens": dict_default.get_top_tokens(),
        "apply": dict_default.apply(TEXTS, tok_lower),
        "unknown_token_id": int(dict_default.unknown_token_id),
        "end_of_sentence_token_id": int(dict_default.end_of_sentence_token_id),
        "min_unused_token_id": int(dict_default.min_unused_token_id),
        "get_token_0": dict_default.get_token(0),
        "get_tokens_0_1_2": dict_default.get_tokens([0, 1, 2]),
        "single_string_apply": dict_default.apply(TEXTS[0], tok_lower),
    }

    # --- Dictionary: unknown_token_policy ---
    dict_small = Dictionary(occurence_lower_bound=0)
    dict_small.fit(["cat dog bird", "cat fish"], Tokenizer())
    unknown_token_policy = {
        "insert": dict_small.apply(["cat dog elephant"], Tokenizer(), unknown_token_policy="Insert"),
        "skip": dict_small.apply(["cat dog elephant"], Tokenizer(), unknown_token_policy="Skip"),
    }

    # --- Dictionary: end_of_sentence_policy = "Insert" ---
    dict_eos = Dictionary(occurence_lower_bound=0, end_of_sentence_policy="Insert")
    dict_eos.fit(["cat dog bird", "cat fish"], Tokenizer())
    eos_apply = dict_eos.apply(["cat dog"], Tokenizer())

    # --- Dictionary: max_dictionary_size ---
    dict_maxsize = Dictionary(occurence_lower_bound=0, max_dictionary_size=2)
    dict_maxsize.fit(["cat dog bird", "cat fish cat"], Tokenizer())
    max_dictionary_size = {
        "top_tokens": dict_maxsize.get_top_tokens(),
        "size": int(dict_maxsize.size),
    }

    # --- Dictionary: fit without a tokenizer (each doc == one token) ---
    dict_no_tokenizer = Dictionary(occurence_lower_bound=0)
    dict_no_tokenizer.fit(["hello", "world", "hello"])
    no_tokenizer_fit = {
        "top_tokens": dict_no_tokenizer.get_top_tokens(),
        "size": int(dict_no_tokenizer.size),
    }

    # --- Dictionary: fit from pre-tokenized (2D) data ---
    dict_pretokenized = Dictionary(occurence_lower_bound=0)
    dict_pretokenized.fit([["a", "b"], ["a", "c"]])
    pretokenized_fit = {
        "top_tokens": dict_pretokenized.get_top_tokens(),
    }

    # --- Dictionary: save/load round trip via the on-disk file format ---
    dict_default.save(SAVED_DICT_PATH)
    dict_reloaded = Dictionary()
    dict_reloaded.load(SAVED_DICT_PATH)
    save_load_round_trip = {
        "size": int(dict_reloaded.size),
        "top_tokens": dict_reloaded.get_top_tokens(),
        "apply": dict_reloaded.apply(TEXTS, tok_lower),
    }
    with open(SAVED_DICT_PATH) as f:
        saved_dictionary_file_contents = f.read()

    fixture = {
        "catboost_version": catboost.__version__,
        "inputs": {
            "texts": TEXTS,
            "number_string": NUMBER_STRING,
        },
        "expected": {
            "tokenize_default": tokenize_default,
            "tokenize_lower": tokenize_lower,
            "tokenize_number_skip": tokenize_number_skip,
            "tokenize_number_replace": tokenize_number_replace,
            "tokenize_number_replace_custom": tokenize_number_replace_custom,
            "tokenize_split_by_set": tokenize_split_by_set,
            "tokenize_no_skip_empty": tokenize_no_skip_empty,
            "tokenize_multi_delimiter": tokenize_multi_delimiter,
            "dictionary_default": dictionary_default,
            "unknown_token_policy": unknown_token_policy,
            "eos_apply": eos_apply,
            "max_dictionary_size": max_dictionary_size,
            "no_tokenizer_fit": no_tokenizer_fit,
            "pretokenized_fit": pretokenized_fit,
            "save_load_round_trip": save_load_round_trip,
            "saved_dictionary_file_contents": saved_dictionary_file_contents,
        },
    }

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"dictionary_default={dictionary_default}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
