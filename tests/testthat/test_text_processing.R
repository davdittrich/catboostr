context("test_text_processing.R")

# P3.6 differential test: the R catboost.Tokenizer/catboost.Dictionary port
# is a pure-R reimplementation (see R/text_processing.R header) of the
# pinned Python catboost==1.2.10 catboost.text_processing.Tokenizer/
# Dictionary (tools/oracle/gen_text_processing_fixture.py). This asserts the
# observable tokenize()/fit()/apply()/get_top_tokens()/save()/load() output
# matches what the fixture oracle script recorded
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

test_that("Tokenizer: 'BySense' separator is explicitly rejected, not silently approximated", {
  expect_error(catboost.Tokenizer(separator_type = "BySense"), "not supported")
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
