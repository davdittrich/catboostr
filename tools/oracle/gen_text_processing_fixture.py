#!/usr/bin/env python3
"""Generate text_processing oracle fixture (catboost-8z4.43 / P3.6, extended
by catboost-8z4.48): deterministic Tokenizer/Dictionary inputs + pinned
Python catboost==1.2.10 catboost.text_processing.Tokenizer/Dictionary
observable outputs, covering the default configuration (Word-level, unigram,
FrequencyBased, ByDelimiter separator) plus number-processing policies,
delimiter modes, unknown-token policy, end-of-sentence policy,
max_dictionary_size, save/load round-trip, fitting without a tokenizer /
pre-tokenized data, and (catboost-8z4.48) the five configurations the
pure-R port could not reproduce and the native tokenizer/dictionary bridge
now closes: BySense tokenization; lemmatizing/token_types/sub_tokens_policy/
languages options; Letter-level dictionaries; multigram (gram_order > 1)
dictionaries; and the Bpe dictionary type.

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
    "Cats beautiful animals.",
    "Dogs loyal, dogs great.",
    "The Cat sat on mat, cat purred.",
]

NUMBER_STRING = "I 12 cats 3 dogs in 2024"


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
    tok_split_by_set = Tokenizer(separator_type="ByDelimiter", delimiter=",;", split_by_set=True)
    tokenize_split_by_set = tok_split_by_set.tokenize("a,b;c,,d")

    # --- Tokenizer: skip_empty = False ---
    tok_no_skip_empty = Tokenizer(delimiter=" ", skip_empty=False)
    tokenize_no_skip_empty = tok_no_skip_empty.tokenize("a  b   c")

    # --- Tokenizer: multi-char delimiter ---
    tok_multi_delimiter = Tokenizer(delimiter="::")
    tokenize_multi_delimiter = tok_multi_delimiter.tokenize("a::b::c")

    # --- Tokenizer: BySense separator (catboost-8z4.48 scope cut 1) ---
    tok_bysense = Tokenizer(separator_type="BySense")
    tokenize_bysense = [tok_bysense.tokenize(t) for t in TEXTS]
    bysense_with_types = tok_bysense.tokenize(TEXTS[2], types=True)
    tokenize_bysense_types = {
        "tokens": [t for t, _ in bysense_with_types],
        "types": [ty for _, ty in bysense_with_types],
    }

    # --- Tokenizer: token_types / sub_tokens_policy (catboost-8z4.48 scope
    #     cut 2). NOTE: `lemmatizing=True` (and, transitively, `languages=`,
    #     which only takes effect together with lemmatizing) is EXCLUDED
    #     from this fixture: the pinned catboost==1.2.10 OSS build's
    #     "Trivial" lemmer implementation is a stub that calls
    #     Y_ENSURE(false) -- vendor/catboost/library/cpp/text_processing/
    #     tokenizer/tokenizer.cpp:267, "Lemmer isn't implemented yet" --
    #     which raises an *uncaught* C++ exception that aborts the whole
    #     Python process (not a catchable Python exception), confirmed by
    #     running this exact script with lemmatizing=True. This is an
    #     upstream OSS-build limitation of the vendor code Python itself
    #     also links, not an R-side gap: there is no Python oracle output to
    #     pin because the Python side crashes identically. See
    #     docs/phase-3/catboost-8z4.48-report.md.
    tok_token_types = Tokenizer(separator_type="BySense", token_types=["Word"])
    tokenize_token_types_word_only = tok_token_types.tokenize(TEXTS[2])

    tok_sub_tokens = Tokenizer(separator_type="BySense", sub_tokens_policy="SeveralTokens")
    tokenize_sub_tokens_several = tok_sub_tokens.tokenize("U.S.A. cats")

    # --- Dictionary: fit/apply/introspection ---
    dict_default = Dictionary(occurence_lower_bound=0)
    dict_default.fit(TEXTS, Tokenizer(lowercasing=True, separator_type="ByDelimiter", delimiter=" "))
    tok_default_lower = Tokenizer(lowercasing=True, separator_type="ByDelimiter", delimiter=" ")
    dictionary_default = {
        "size": int(dict_default.size),
        "top_tokens": dict_default.get_top_tokens(),
        "apply": dict_default.apply(TEXTS, tok_default_lower),
        "unknown_token_id": int(dict_default.unknown_token_id),
        "end_of_sentence_token_id": int(dict_default.end_of_sentence_token_id),
        "min_unused_token_id": int(dict_default.min_unused_token_id),
        "get_token_0": dict_default.get_token(0),
        "get_tokens_0_1_2": dict_default.get_tokens([0, 1, 2]),
        "single_string_apply": dict_default.apply(TEXTS[0], tok_default_lower),
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

    # --- Dictionary: Letter-level tokenization (catboost-8z4.48 scope cut 3) ---
    dict_letter = Dictionary(token_level_type="Letter", occurence_lower_bound=0)
    dict_letter.fit(["cat", "cats"], Tokenizer())
    letter_level = {
        "top_tokens": dict_letter.get_top_tokens(),
        "size": int(dict_letter.size),
        "apply": dict_letter.apply(["cat"], Tokenizer()),
    }

    # --- Dictionary: multigram gram_order > 1 (catboost-8z4.48 scope cut 4) ---
    dict_multigram = Dictionary(gram_order=2, occurence_lower_bound=0)
    dict_multigram.fit(["cat dog bird", "cat dog fish"], Tokenizer())
    multigram = {
        "top_tokens": dict_multigram.get_top_tokens(),
        "size": int(dict_multigram.size),
        "apply": dict_multigram.apply(["cat dog bird"], Tokenizer()),
    }

    # --- Dictionary: Bpe dictionary type (catboost-8z4.48 scope cut 5).
    # NOTE: Python's Dictionary.fit() only supports Bpe fitting from a file
    # path (_text_processing.pxi's __fit_bpe raises "Now you can fit
    # dictionary from file." for array-like `data`, confirmed by running
    # this exact script against array data) -- an artificial restriction of
    # Python's convenience wrapper, not of the underlying vendor C++
    # (TBpeDictionaryBuilder::Add() is a normal incremental in-memory API).
    # R's native bridge supports fitting Bpe from in-memory data directly,
    # which is strictly *more* capable than Python here. To still pin a
    # genuine Python oracle value, the same corpus is written to a file and
    # fit that way (with the same default ByDelimiter Tokenizer R's
    # in-memory-array test will apply itself); since both paths run the
    # identical two-pass alphabet-then-merge algorithm
    # (library/cpp/text_processing/app_helpers/app_helpers.cpp's
    # BuildBpeWord) over the same token stream, the resulting merges are
    # expected to be identical regardless of the file/array origin of that
    # stream.
    BPE_CORPUS_PATH = os.path.join(FIXTURE_DIR, "text_processing_bpe_corpus.txt")
    with open(BPE_CORPUS_PATH, "w") as f:
        f.write("cat cats catfish\ndog dogs\n")
    dict_bpe = Dictionary(dictionary_type="Bpe", occurence_lower_bound=0, num_bpe_units=5)
    dict_bpe.fit(BPE_CORPUS_PATH, Tokenizer())
    # NOTE: get_top_tokens()/single-token apply() are unimplemented for
    # TBpeDictionary in vendor C++ itself (Y_ENSURE(false, ...) in
    # library/cpp/text_processing/dictionary/bpe_dictionary.cpp:112,175,
    # confirmed by running this exact script) -- both Python and R raise
    # identically because both call the same vendor method.
    bpe_size = int(dict_bpe.size)
    bpe = {
        "size": bpe_size,
        "apply": dict_bpe.apply(["cat cats catfish dog dogs"], Tokenizer()),
        "tokens_by_id": [dict_bpe.get_token(i) for i in range(bpe_size)],
    }
    os.remove(BPE_CORPUS_PATH)

    # --- Dictionary: save/load round trip via on-disk file format ---
    dict_default.save(SAVED_DICT_PATH)
    dict_reloaded = Dictionary()
    dict_reloaded.load(SAVED_DICT_PATH)
    save_load_round_trip = {
        "size": int(dict_reloaded.size),
        "top_tokens": dict_reloaded.get_top_tokens(),
        "apply": dict_reloaded.apply(TEXTS, tok_default_lower),
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
            "tokenize_bysense": tokenize_bysense,
            "tokenize_bysense_types": tokenize_bysense_types,
            "tokenize_token_types_word_only": tokenize_token_types_word_only,
            "tokenize_sub_tokens_several": tokenize_sub_tokens_several,
            "dictionary_default": dictionary_default,
            "unknown_token_policy": unknown_token_policy,
            "eos_apply": eos_apply,
            "max_dictionary_size": max_dictionary_size,
            "no_tokenizer_fit": no_tokenizer_fit,
            "pretokenized_fit": pretokenized_fit,
            "letter_level": letter_level,
            "multigram": multigram,
            "bpe": bpe,
            "save_load_round_trip": save_load_round_trip,
            "saved_dictionary_file_contents": saved_dictionary_file_contents,
        },
    }

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"Wrote {FIXTURE_PATH}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
