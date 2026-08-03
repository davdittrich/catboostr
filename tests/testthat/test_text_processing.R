context("test_text_processing.R")

# P3.6 / catboost-8z4.48 differential test: catboost.Tokenizer/
# catboost.Dictionary (R/text_processing.R) are native bridges around the
# same vendor C++ (NTextProcessing::NTokenizer::TTokenizer,
# NTextProcessing::NDictionary::TDictionary/TDictionaryBuilder/
# TBpeDictionary/TBpeDictionaryBuilder) the pinned Python catboost==1.2.10
# catboost.text_processing.Tokenizer/Dictionary (tools/oracle/
# gen_text_processing_fixture.py) links -- so R and Python are expected to
# be byte-identical, not merely "close". This asserts the observable
# tokenize()/fit()/apply()/get_top_tokens()/save()/load() output matches
# what the fixture oracle script recorded
# (tests/fixtures/oracle/text_processing.json), plus a genuine file-format
# round trip against a dictionary saved directly by the pinned oracle.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_text_processing_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "text_processing.json"),
  simplifyVector = TRUE
)
expected <- fixture$expected
texts <- fixture$inputs$texts
number_string <- fixture$inputs$number_string

test_that("Tokenizer: default (ByDelimiter, delimiter=' ') matches Python oracle", {
  tok <- catboost.Tokenizer()
  for (i in seq_along(texts)) {
    expect_equal(catboost.tokenizer.tokenize(tok, texts[i]), expected$tokenize_default[[i]])
  }
})

test_that("Tokenizer: lowercasing matches Python oracle", {
  tok <- catboost.Tokenizer(lowercasing = TRUE, separator_type = "ByDelimiter", delimiter = " ")
  for (i in seq_along(texts)) {
    expect_equal(catboost.tokenizer.tokenize(tok, texts[i]), expected$tokenize_lower[[i]])
  }
})

test_that("Tokenizer: number_process_policy matches Python oracle", {
  expect_equal(
    catboost.tokenizer.tokenize(catboost.Tokenizer(number_process_policy = "Skip"), number_string),
    expected$tokenize_number_skip
  )
  expect_equal(
    catboost.tokenizer.tokenize(catboost.Tokenizer(number_process_policy = "Replace"), number_string),
    expected$tokenize_number_replace
  )
  expect_equal(
    catboost.tokenizer.tokenize(
      catboost.Tokenizer(number_process_policy = "Replace", number_token = "<NUM>"), number_string
    ),
    expected$tokenize_number_replace_custom
  )
})

test_that("Tokenizer: split_by_set matches Python oracle", {
  tok <- catboost.Tokenizer(separator_type = "ByDelimiter", delimiter = ",;", split_by_set = TRUE)
  expect_equal(catboost.tokenizer.tokenize(tok, "a,b;c,,d"), expected$tokenize_split_by_set)
})

test_that("Tokenizer: skip_empty = FALSE matches Python oracle", {
  tok <- catboost.Tokenizer(delimiter = " ", skip_empty = FALSE)
  expect_equal(catboost.tokenizer.tokenize(tok, "a  b   c"), expected$tokenize_no_skip_empty)
})

test_that("Tokenizer: multi-character delimiter matches Python oracle", {
  tok <- catboost.Tokenizer(delimiter = "::")
  expect_equal(catboost.tokenizer.tokenize(tok, "a::b::c"), expected$tokenize_multi_delimiter)
})

# --- catboost-8z4.48 scope cut 1: BySense tokenization -------------------

test_that("Tokenizer: 'BySense' separator matches Python oracle (catboost-8z4.48)", {
  tok <- catboost.Tokenizer(separator_type = "BySense")
  for (i in seq_along(texts)) {
    expect_equal(catboost.tokenizer.tokenize(tok, texts[i]), expected$tokenize_bysense[[i]])
  }
  with_types <- catboost.tokenizer.tokenize(tok, texts[3], types = TRUE)
  expect_equal(with_types$token, expected$tokenize_bysense_types$tokens)
  expect_equal(with_types$type, expected$tokenize_bysense_types$types)
})

# --- catboost-8z4.48 scope cut 2: token_types / sub_tokens_policy --------
# (lemmatizing is excluded from oracle parity: the pinned catboost==1.2.10
# OSS build's Lemmer implementation is an unimplemented stub --
# vendor/catboost/library/cpp/text_processing/tokenizer/tokenizer.cpp:267,
# confirmed while generating the fixture -- so there is no Python oracle
# output to pin. catboost.Tokenizer(lemmatizing = TRUE) reaches that
# identical vendor Y_ENSURE via TTokenizer's constructor, which
# src/catboostr.cpp's CatBoostTextTokenizerCreate_R runs inside
# R_API_BEGIN()/R_API_END(); R_API_END() catches std::exception (yexception
# is a subclass) and converts it to a normal, catchable R error(), so the
# failure is asserted below with expect_error() rather than skipped.)

test_that("Tokenizer: lemmatizing = TRUE surfaces the vendor 'not implemented' error", {
  expect_error(catboost.Tokenizer(lemmatizing = TRUE), "Lemmer isn't implemented yet")
})

test_that("Tokenizer: token_types filters to the requested types (BySense) matches Python oracle", {
  tok <- catboost.Tokenizer(separator_type = "BySense", token_types = "Word")
  expect_equal(catboost.tokenizer.tokenize(tok, texts[3]), expected$tokenize_token_types_word_only)
})

test_that("Tokenizer: sub_tokens_policy = 'SeveralTokens' matches Python oracle", {
  tok <- catboost.Tokenizer(separator_type = "BySense", sub_tokens_policy = "SeveralTokens")
  expect_equal(catboost.tokenizer.tokenize(tok, "U.S.A. cats"), expected$tokenize_sub_tokens_several)
})

test_that("Dictionary: fit/apply/introspection matches Python oracle", {
  tok <- catboost.Tokenizer(lowercasing = TRUE, separator_type = "ByDelimiter", delimiter = " ")
  d <- catboost.Dictionary(occurence_lower_bound = 0)
  catboost.dictionary.fit(d, texts, tok)

  exp <- expected$dictionary_default
  expect_equal(catboost.dictionary.size(d), exp$size)
  expect_equal(catboost.dictionary.get_top_tokens(d), exp$top_tokens)
  expect_equal(catboost.dictionary.apply(d, texts, tok), lapply(exp$apply, as.integer))
  expect_equal(catboost.dictionary.unknown_token_id(d), exp$unknown_token_id)
  expect_equal(catboost.dictionary.end_of_sentence_token_id(d), exp$end_of_sentence_token_id)
  expect_equal(catboost.dictionary.min_unused_token_id(d), exp$min_unused_token_id)
  expect_equal(catboost.dictionary.get_token(d, 0), exp$get_token_0)
  expect_equal(catboost.dictionary.get_tokens(d, c(0, 1, 2)), exp$get_tokens_0_1_2)
  expect_equal(catboost.dictionary.apply(d, texts[1], tok), as.integer(exp$single_string_apply))
})

test_that("Dictionary: unknown_token_policy matches Python oracle", {
  d <- catboost.Dictionary(occurence_lower_bound = 0)
  catboost.dictionary.fit(d, c("cat dog bird", "cat fish"), catboost.Tokenizer())

  expect_equal(
    catboost.dictionary.apply(d, list("cat dog elephant"), catboost.Tokenizer(), unknown_token_policy = "Insert"),
    list(as.integer(expected$unknown_token_policy$insert))
  )
  expect_equal(
    catboost.dictionary.apply(d, list("cat dog elephant"), catboost.Tokenizer(), unknown_token_policy = "Skip"),
    list(as.integer(expected$unknown_token_policy$skip))
  )
})

test_that("Dictionary: end_of_sentence_policy = 'Insert' matches Python oracle", {
  d <- catboost.Dictionary(occurence_lower_bound = 0, end_of_sentence_policy = "Insert")
  catboost.dictionary.fit(d, c("cat dog bird", "cat fish"), catboost.Tokenizer())
  expect_equal(
    catboost.dictionary.apply(d, list("cat dog"), catboost.Tokenizer()),
    list(as.integer(expected$eos_apply))
  )
})

test_that("Dictionary: max_dictionary_size matches Python oracle", {
  d <- catboost.Dictionary(occurence_lower_bound = 0, max_dictionary_size = 2)
  catboost.dictionary.fit(d, c("cat dog bird", "cat fish cat"), catboost.Tokenizer())
  expect_equal(catboost.dictionary.get_top_tokens(d), expected$max_dictionary_size$top_tokens)
  expect_equal(catboost.dictionary.size(d), expected$max_dictionary_size$size)
})

test_that("Dictionary: fit without a tokenizer matches Python oracle", {
  d <- catboost.Dictionary(occurence_lower_bound = 0)
  catboost.dictionary.fit(d, c("hello", "world", "hello"))
  expect_equal(catboost.dictionary.get_top_tokens(d), expected$no_tokenizer_fit$top_tokens)
  expect_equal(catboost.dictionary.size(d), expected$no_tokenizer_fit$size)
})

test_that("Dictionary: fit from pre-tokenized data matches Python oracle", {
  d <- catboost.Dictionary(occurence_lower_bound = 0)
  catboost.dictionary.fit(d, list(c("a", "b"), c("a", "c")))
  expect_equal(catboost.dictionary.get_top_tokens(d), expected$pretokenized_fit$top_tokens)
})

# --- catboost-8z4.48 scope cut 3: Letter-level dictionaries --------------

test_that("Dictionary: token_level_type = 'Letter' matches Python oracle (catboost-8z4.48)", {
  d <- catboost.Dictionary(token_level_type = "Letter", occurence_lower_bound = 0)
  catboost.dictionary.fit(d, c("cat", "cats"), catboost.Tokenizer())

  exp <- expected$letter_level
  expect_equal(catboost.dictionary.get_top_tokens(d), exp$top_tokens)
  expect_equal(catboost.dictionary.size(d), exp$size)
  expect_equal(
    catboost.dictionary.apply(d, list("cat"), catboost.Tokenizer()),
    list(as.integer(unlist(exp$apply)))
  )
})

# --- catboost-8z4.48 scope cut 4: multigram (gram_order > 1) -------------

test_that("Dictionary: gram_order > 1 (multigram) matches Python oracle (catboost-8z4.48)", {
  d <- catboost.Dictionary(gram_order = 2, occurence_lower_bound = 0)
  catboost.dictionary.fit(d, c("cat dog bird", "cat dog fish"), catboost.Tokenizer())

  exp <- expected$multigram
  expect_equal(catboost.dictionary.get_top_tokens(d), exp$top_tokens)
  expect_equal(catboost.dictionary.size(d), exp$size)
  expect_equal(
    catboost.dictionary.apply(d, list("cat dog bird"), catboost.Tokenizer()),
    list(as.integer(unlist(exp$apply)))
  )
})

# --- catboost-8z4.48 scope cut 5: Bpe dictionary type --------------------
#
# Python's Dictionary.fit() only supports fitting a Bpe dictionary from a
# file path (_text_processing.pxi's __fit_bpe raises "Now you can fit
# dictionary from file." for array-like data -- confirmed while generating
# the fixture); the oracle was generated by writing this exact corpus to a
# file and fitting from there with the default (ByDelimiter, delimiter=" ")
# Tokenizer. R's native bridge is not limited to file input: it builds the
# same alphabet-then-merge corpus in memory from data tokenized R-side with
# that same default Tokenizer, so the resulting merges are expected to be
# byte-identical (both call the same vendor
# TDictionaryBuilder/TBpeDictionaryBuilder over the same token stream).
# get_top_tokens()/single-token apply() are unimplemented for TBpeDictionary
# in vendor C++ itself (Y_ENSURE(false, ...) in
# library/cpp/text_processing/dictionary/bpe_dictionary.cpp), so this test
# does not call them.

test_that("Dictionary: dictionary_type = 'Bpe' matches Python oracle (catboost-8z4.48)", {
  d <- catboost.Dictionary(dictionary_type = "Bpe", occurence_lower_bound = 0, num_bpe_units = 5)
  catboost.dictionary.fit(d, c("cat cats catfish", "dog dogs"), catboost.Tokenizer())

  exp <- expected$bpe
  expect_equal(catboost.dictionary.size(d), exp$size)
  expect_equal(
    catboost.dictionary.apply(d, list("cat cats catfish dog dogs"), catboost.Tokenizer()),
    list(as.integer(unlist(exp$apply)))
  )
  expect_equal(
    catboost.dictionary.get_tokens(d, seq_len(exp$size) - 1L),
    exp$tokens_by_id
  )
})

test_that("Dictionary: save() writes the exact format the Python oracle wrote", {
  tok <- catboost.Tokenizer(lowercasing = TRUE, separator_type = "ByDelimiter", delimiter = " ")
  d <- catboost.Dictionary(occurence_lower_bound = 0)
  catboost.dictionary.fit(d, texts, tok)

  path <- tempfile()
  on.exit(unlink(path))
  catboost.dictionary.save(d, path)
  actual_lines <- readLines(path)
  expected_lines <- strsplit(expected$saved_dictionary_file_contents, "\n")[[1]]

  # The JSON header's key order is not a contract (both R's jsonlite and
  # Python's json module parse either order identically, and each tool can
  # load a dictionary file written by the other -- see the load() test
  # below and the reverse check in docs/phase-3/catboost-8z4.43-report.md).
  # Compare the header as parsed JSON and the data lines byte-for-byte.
  sort_by_name <- function(x) x[order(names(x))]
  expect_equal(sort_by_name(jsonlite::fromJSON(actual_lines[1])), sort_by_name(jsonlite::fromJSON(expected_lines[1])))
  expect_equal(actual_lines[-1], expected_lines[-1])
})

test_that("Dictionary: load() reads a dictionary file saved directly by the Python oracle", {
  tok <- catboost.Tokenizer(lowercasing = TRUE, separator_type = "ByDelimiter", delimiter = " ")
  d <- catboost.Dictionary()
  catboost.dictionary.load(d, testthat::test_path("..", "fixtures", "oracle", "text_processing_dictionary.freq"))

  exp <- expected$save_load_round_trip
  expect_equal(catboost.dictionary.size(d), exp$size)
  expect_equal(catboost.dictionary.get_top_tokens(d), exp$top_tokens)
  expect_equal(catboost.dictionary.apply(d, texts, tok), lapply(exp$apply, as.integer))
})
