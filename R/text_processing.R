#' @import jsonlite
NULL

# ---------------------------------------------------------------------------
# catboost.Tokenizer / catboost.Dictionary
#
# Native bridges (catboost-8z4.48) around Python's
# catboost.text_processing.Tokenizer / catboost.text_processing.Dictionary.
# Both Python classes are thin Cython wrappers
# (vendor/catboost/catboost/python-package/catboost/_text_processing.pxi)
# around vendor/catboost/library/cpp/text_processing/{tokenizer,dictionary},
# which src/CMakeLists.txt now links directly into catboostr (same linking
# pattern P3.3 established for dataset-statistics; see src/catboostr.cpp's
# CatBoostText*_R bridges). Every option Python exposes -- BySense
# separation, lemmatizing/token_types/sub_tokens_policy/languages,
# Letter-level and multigram (gram_order > 1) dictionaries, and the Bpe
# dictionary type -- is backed by the identical vendor C++ code the pinned
# Python oracle runs, so R and Python are byte-identical by construction
# rather than by a hand-verified pure-R reimplementation (the earlier
# ~643-line pure-R port this replaces only covered each option's default
# value and explicitly rejected the rest).
# ---------------------------------------------------------------------------

.catboost.tp.tokenize_line <- function(line, tokenizer) {
  # Mirrors _text_processing.pxi: a bare (length-1) string is tokenized (if a
  # tokenizer is given) or treated as a single token (if not); an
  # already-tokenized character vector/list element is used as-is.
  if (is.character(line) && length(line) == 1 && !is.null(tokenizer)) {
    catboost.tokenizer.tokenize(tokenizer, line)
  } else {
    as.character(line)
  }
}

#' @name catboost.Tokenizer
#' @title Create text Tokenizer
#'
#' @description Splits a string into tokens ahead of \code{catboost.Dictionary}
#' fitting/application. Native bridge to Python's
#' \code{catboost.text_processing.Tokenizer}, backed directly by vendor's
#' \code{NTextProcessing::NTokenizer::TTokenizer}.
#'
#' @param lowercasing Lowercase each token. Default value: FALSE
#' @param number_process_policy One of "Skip", "LeaveAsIs", "Replace".
#'
#' Default value: "LeaveAsIs"
#' @param number_token Replacement token used when \code{number_process_policy}
#' is "Replace".
#'
#' Default value: "🔢" (matches vendor default)
#' @param separator_type Tokenization method: "ByDelimiter" or "BySense".
#'
#' Default value: "ByDelimiter"
#' @param delimiter Delimiter string used to split tokens (ByDelimiter mode).
#'
#' Default value: " "
#' @param split_by_set If TRUE, each individual character in \code{delimiter}
#' is treated as its own delimiter (ByDelimiter mode).
#'
#' Default value: FALSE
#' @param skip_empty Skip empty tokens adjacent to delimiters (ByDelimiter mode).
#'
#' Default value: TRUE
#' @param token_types Character vector of token types kept after
#' tokenization (BySense mode). Possible values: "Word", "Digit",
#' "Punctuation", "SentenceBreak", "ParagraphBreak", "Unknown".
#'
#' Default value: NULL (vendor default: c("Word", "Digit", "Unknown"))
#' @param sub_tokens_policy Subtoken processing policy (BySense mode): one of
#' "SingleToken", "SeveralTokens".
#'
#' Default value: "SingleToken"
#' @param languages Character vector of language names used for
#' lemmatizing/BySense tokenization.
#'
#' Default value: NULL (all languages)
#' @param lemmatizing Apply lemmatization to tokens.
#'
#' Default value: FALSE
#' @return catboost.Tokenizer
#' @export
catboost.Tokenizer <- function(lowercasing = NULL,
                                number_process_policy = NULL,
                                number_token = NULL,
                                separator_type = NULL,
                                delimiter = NULL,
                                split_by_set = NULL,
                                skip_empty = NULL,
                                token_types = NULL,
                                sub_tokens_policy = NULL,
                                languages = NULL,
                                lemmatizing = NULL) {
  lowercasing <- if (is.null(lowercasing)) FALSE else lowercasing
  lemmatizing <- if (is.null(lemmatizing)) FALSE else lemmatizing
  number_process_policy <- if (is.null(number_process_policy)) "LeaveAsIs" else number_process_policy
  number_token <- if (is.null(number_token)) "\U0001F522" else number_token
  separator_type <- if (is.null(separator_type)) "ByDelimiter" else separator_type
  delimiter <- if (is.null(delimiter)) " " else delimiter
  split_by_set <- if (is.null(split_by_set)) FALSE else split_by_set
  skip_empty <- if (is.null(skip_empty)) TRUE else skip_empty
  sub_tokens_policy <- if (is.null(sub_tokens_policy)) "SingleToken" else sub_tokens_policy

  handle <- .Call(
    "CatBoostTextTokenizerCreate_R",
    lowercasing, lemmatizing, number_process_policy, number_token,
    separator_type, delimiter, split_by_set, skip_empty,
    token_types, sub_tokens_policy, languages
  )

  structure(
    list(
      handle = handle,
      lowercasing = lowercasing,
      lemmatizing = lemmatizing,
      number_process_policy = number_process_policy,
      number_token = number_token,
      separator_type = separator_type,
      delimiter = delimiter,
      split_by_set = split_by_set,
      skip_empty = skip_empty,
      token_types = token_types,
      sub_tokens_policy = sub_tokens_policy,
      languages = languages
    ),
    class = "catboost.Tokenizer"
  )
}


#' @name catboost.tokenizer.tokenize
#' @title Tokenize a string
#'
#' @description Split \code{string} into tokens using a \code{catboost.Tokenizer}.
#'
#' @param tokenizer A catboost.Tokenizer object.
#'
#' Default value: Required argument
#' @param string Input string.
#'
#' Default value: Required argument
#' @param types If TRUE, also return token types.
#'
#' Default value: FALSE
#' @return A character vector of tokens, or (if \code{types = TRUE}) a
#' data.frame with \code{token} and \code{type} columns.
#' @export
catboost.tokenizer.tokenize <- function(tokenizer, string, types = FALSE) {
  if (!inherits(tokenizer, "catboost.Tokenizer")) {
    stop("catboost.tokenizer.tokenize: 'tokenizer' must be a catboost.Tokenizer object.")
  }
  result <- .Call("CatBoostTextTokenizerTokenize_R", tokenizer$handle, as.character(string))
  if (types) {
    return(data.frame(token = result$tokens, type = result$types, stringsAsFactors = FALSE))
  }
  result$tokens
}


.catboost.tp.dictionary_defaults <- function() {
  list(
    token_level_type = "Word",
    gram_order = 1L,
    skip_step = 0L,
    start_token_id = 0L,
    end_of_word_policy = "Insert",
    end_of_sentence_policy = "Skip",
    occurence_lower_bound = 50L,
    max_dictionary_size = -1L,
    dictionary_type = "FrequencyBased",
    num_bpe_units = 0L,
    skip_unknown = FALSE
  )
}

#' @name catboost.Dictionary
#' @title Create text Dictionary
#'
#' @description Builds a token/id vocabulary from text data. Native bridge to
#' Python's \code{catboost.text_processing.Dictionary}, backed directly by
#' vendor's \code{NTextProcessing::NDictionary::TDictionary}/
#' \code{TDictionaryBuilder}/\code{TBpeDictionary}/\code{TBpeDictionaryBuilder}.
#' Use \code{catboost.dictionary.fit} to train it and
#' \code{catboost.dictionary.apply} to convert text to token ids.
#'
#' @param token_level_type "Word" or "Letter".
#'
#' Default value: "Word"
#' @param gram_order The number of words/letters joined into each token
#' (multigrams).
#'
#' Default value: 1
#' @param start_token_id Initial shift for assigned token identifiers.
#'
#' Default value: 0
#' @param end_of_word_policy "Skip" or "Insert"; used with Letter-level
#' dictionaries.
#'
#' Default value: "Insert"
#' @param end_of_sentence_policy "Skip" or "Insert"; whether
#' \code{catboost.dictionary.apply} appends an end-of-sentence token id
#' after a tokenized line.
#'
#' Default value: "Skip"
#' @param occurence_lower_bound Minimum corpus occurrence count for a token
#' to be kept in the dictionary.
#'
#' Default value: 50
#' @param max_dictionary_size Maximum number of tokens to keep, or -1 for
#' unlimited.
#'
#' Default value: -1
#' @param dictionary_type "FrequencyBased" or "Bpe".
#'
#' Default value: "FrequencyBased"
#' @param num_bpe_units Number of token-pair merges to perform (Bpe type).
#'
#' Default value: 0
#' @param skip_unknown Skip unknown tokens when building a Bpe dictionary.
#'
#' Default value: FALSE
#' @param skip_step Number of words/letters skipped when joining them into
#' tokens; only takes effect when \code{gram_order > 1}.
#'
#' Default value: 0
#' @return A mutable catboost.Dictionary object (\code{fit}/\code{load}
#' update it in place).
#' @export
catboost.Dictionary <- function(token_level_type = NULL,
                                 gram_order = NULL,
                                 start_token_id = NULL,
                                 end_of_word_policy = NULL,
                                 end_of_sentence_policy = NULL,
                                 occurence_lower_bound = NULL,
                                 max_dictionary_size = NULL,
                                 dictionary_type = NULL,
                                 num_bpe_units = NULL,
                                 skip_unknown = NULL,
                                 skip_step = NULL) {
  defaults <- .catboost.tp.dictionary_defaults()

  env <- new.env(parent = emptyenv())
  env$token_level_type <- if (is.null(token_level_type)) defaults$token_level_type else token_level_type
  env$gram_order <- if (is.null(gram_order)) defaults$gram_order else as.integer(gram_order)
  env$skip_step <- if (is.null(skip_step)) defaults$skip_step else as.integer(skip_step)
  env$start_token_id <- if (is.null(start_token_id)) defaults$start_token_id else as.integer(start_token_id)
  env$end_of_word_policy <- if (is.null(end_of_word_policy)) defaults$end_of_word_policy else end_of_word_policy
  env$end_of_sentence_policy <-
    if (is.null(end_of_sentence_policy)) defaults$end_of_sentence_policy else end_of_sentence_policy
  env$occurence_lower_bound <-
    if (is.null(occurence_lower_bound)) defaults$occurence_lower_bound else as.integer(occurence_lower_bound)
  env$max_dictionary_size <-
    if (is.null(max_dictionary_size)) defaults$max_dictionary_size else as.integer(max_dictionary_size)
  env$dictionary_type <- if (is.null(dictionary_type)) defaults$dictionary_type else dictionary_type
  env$num_bpe_units <- if (is.null(num_bpe_units)) defaults$num_bpe_units else as.integer(num_bpe_units)
  env$skip_unknown <- if (is.null(skip_unknown)) defaults$skip_unknown else skip_unknown

  env$initialized <- FALSE
  env$handle <- NULL

  structure(env, class = "catboost.Dictionary")
}

.catboost.dictionary.check_initialized <- function(dictionary) {
  if (!isTRUE(dictionary$initialized)) {
    stop(
      "catboost.Dictionary is not initialized yet; call catboost.dictionary.fit ",
      "or catboost.dictionary.load first."
    )
  }
}

#' @name catboost.dictionary.fit
#' @title Train a Dictionary
#'
#' @description Builds the token/id vocabulary from \code{data}.
#'
#' @param dictionary A catboost.Dictionary object.
#'
#' Default value: Required argument
#' @param data A character vector, or a list mixing strings and
#' already-tokenized character vectors.
#'
#' Default value: Required argument
#' @param tokenizer An optional catboost.Tokenizer used to split each
#' length-1 string element of \code{data} into tokens; character vector
#' elements of length > 1 are treated as already tokenized regardless.
#'
#' Default value: NULL
#' @param verbose Unused; accepted for interface parity with the Python API.
#'
#' Default value: FALSE
#' @return The (mutated) catboost.Dictionary, invisibly.
#' @export
catboost.dictionary.fit <- function(dictionary, data, tokenizer = NULL, verbose = FALSE) {
  if (!inherits(dictionary, "catboost.Dictionary")) {
    stop("catboost.dictionary.fit: 'dictionary' must be a catboost.Dictionary object.")
  }
  if (is.character(data)) {
    data <- as.list(data)
  }
  if (!is.list(data)) {
    stop("catboost.dictionary.fit: 'data' must be a character vector or a list of character vectors.")
  }

  lines <- lapply(data, .catboost.tp.tokenize_line, tokenizer = tokenizer)

  dictionary$handle <- .Call(
    "CatBoostTextDictionaryFit_R",
    lines,
    dictionary$token_level_type,
    dictionary$gram_order,
    dictionary$skip_step,
    dictionary$start_token_id,
    dictionary$end_of_word_policy,
    dictionary$end_of_sentence_policy,
    dictionary$occurence_lower_bound,
    dictionary$max_dictionary_size,
    dictionary$dictionary_type,
    dictionary$num_bpe_units,
    dictionary$skip_unknown
  )
  dictionary$initialized <- TRUE
  invisible(dictionary)
}

#' @name catboost.dictionary.apply
#' @title Apply a Dictionary
#'
#' @description Converts \code{data} into token ids.
#'
#' @param dictionary A fitted catboost.Dictionary object.
#'
#' Default value: Required argument
#' @param data A string, character vector, or list mixing strings and
#' already-tokenized character vectors.
#'
#' Default value: Required argument
#' @param tokenizer An optional catboost.Tokenizer, see
#' \code{catboost.dictionary.fit}.
#'
#' Default value: NULL
#' @param unknown_token_policy "Skip" (drop unknown tokens) or "Insert"
#' (emit \code{unknown_token_id}).
#'
#' Default value: "Skip"
#' @return An integer vector of token ids if \code{data} is a single
#' string, otherwise a list of integer vectors (one per document).
#' @export
catboost.dictionary.apply <- function(dictionary, data, tokenizer = NULL, unknown_token_policy = NULL) {
  .catboost.dictionary.check_initialized(dictionary)
  unknown_token_policy <- if (is.null(unknown_token_policy)) "Skip" else unknown_token_policy

  need_to_extract <- is.character(data) && length(data) == 1
  lines <- if (is.character(data)) as.list(data) else data
  if (!is.list(lines)) {
    stop("catboost.dictionary.apply: 'data' must be a string, character vector, or list of character vectors.")
  }

  tokenized <- lapply(lines, .catboost.tp.tokenize_line, tokenizer = tokenizer)
  result <- .Call("CatBoostTextDictionaryApply_R", dictionary$handle, tokenized, unknown_token_policy)

  if (need_to_extract) {
    return(result[[1]])
  }
  result
}

#' @name catboost.dictionary.size
#' @title Dictionary size
#' @param dictionary A fitted catboost.Dictionary object.
#'
#' Default value: Required argument
#' @return An integer.
#' @export
catboost.dictionary.size <- function(dictionary) {
  .catboost.dictionary.check_initialized(dictionary)
  .Call("CatBoostTextDictionarySize_R", dictionary$handle)
}

#' @name catboost.dictionary.get_token
#' @title Get a single token by id
#' @param dictionary A fitted catboost.Dictionary object.
#'
#' Default value: Required argument
#' @param token_id A token id.
#'
#' Default value: Required argument
#' @return A character string.
#' @export
catboost.dictionary.get_token <- function(dictionary, token_id) {
  catboost.dictionary.get_tokens(dictionary, token_id)[[1]]
}

#' @name catboost.dictionary.get_tokens
#' @title Get tokens by id
#' @param dictionary A fitted catboost.Dictionary object.
#'
#' Default value: Required argument
#' @param token_ids A vector of token ids.
#'
#' Default value: Required argument
#' @return A character vector of tokens.
#' @export
catboost.dictionary.get_tokens <- function(dictionary, token_ids) {
  .catboost.dictionary.check_initialized(dictionary)
  .Call("CatBoostTextDictionaryGetTokens_R", dictionary$handle, as.integer(token_ids))
}

#' @name catboost.dictionary.get_top_tokens
#' @title Most frequent tokens
#' @param dictionary A fitted catboost.Dictionary object.
#'
#' Default value: Required argument
#' @param top_size Number of tokens to return.
#'
#' Default value: 10
#' @return A character vector of the most frequent tokens, most frequent first.
#' @export
catboost.dictionary.get_top_tokens <- function(dictionary, top_size = NULL) {
  .catboost.dictionary.check_initialized(dictionary)
  top_size <- if (is.null(top_size)) 10L else as.integer(top_size)
  .Call("CatBoostTextDictionaryGetTopTokens_R", dictionary$handle, top_size)
}

#' @name catboost.dictionary.unknown_token_id
#' @title Identifier of the unknown-token placeholder
#' @param dictionary A fitted catboost.Dictionary object.
#'
#' Default value: Required argument
#' @return An integer id.
#' @export
catboost.dictionary.unknown_token_id <- function(dictionary) {
  .catboost.dictionary.check_initialized(dictionary)
  .Call("CatBoostTextDictionaryUnknownTokenId_R", dictionary$handle)
}

#' @name catboost.dictionary.end_of_sentence_token_id
#' @title Identifier of the end-of-sentence token
#' @param dictionary A fitted catboost.Dictionary object.
#'
#' Default value: Required argument
#' @return An integer id.
#' @export
catboost.dictionary.end_of_sentence_token_id <- function(dictionary) {
  .catboost.dictionary.check_initialized(dictionary)
  .Call("CatBoostTextDictionaryEndOfSentenceTokenId_R", dictionary$handle)
}

#' @name catboost.dictionary.min_unused_token_id
#' @title Smallest unused token identifier
#' @param dictionary A fitted catboost.Dictionary object.
#'
#' Default value: Required argument
#' @return An integer id.
#' @export
catboost.dictionary.min_unused_token_id <- function(dictionary) {
  .catboost.dictionary.check_initialized(dictionary)
  .Call("CatBoostTextDictionaryMinUnusedTokenId_R", dictionary$handle)
}

#' @name catboost.dictionary.save
#' @title Save a Dictionary
#'
#' @description Writes the vendor "id_count_token" text format (a JSON
#' options header line, a token-count line, then one "id\\tcount\\ttoken"
#' line per token) for "FrequencyBased" dictionaries -- byte-identical to
#' the pinned Python oracle's \code{Dictionary.save} because both call the
#' same vendor \code{TDictionary::Save}. "Bpe" dictionaries additionally
#' require \code{bpe_path} and are written via vendor's
#' \code{TBpeDictionary::Save}.
#'
#' @param dictionary A fitted catboost.Dictionary object.
#'
#' Default value: Required argument
#' @param frequency_dict_path Output file path for the frequency-based
#' alphabet.
#'
#' Default value: Required argument
#' @param bpe_path Output file path for Bpe merge data; required when
#' \code{dictionary$dictionary_type == "Bpe"}.
#'
#' Default value: NULL
#' @return The catboost.Dictionary, invisibly.
#' @export
catboost.dictionary.save <- function(dictionary, frequency_dict_path, bpe_path = NULL) {
  .catboost.dictionary.check_initialized(dictionary)
  .Call(
    "CatBoostTextDictionarySave_R",
    dictionary$handle, dictionary$dictionary_type, frequency_dict_path, bpe_path
  )
  invisible(dictionary)
}

#' @name catboost.dictionary.load
#' @title Load a Dictionary
#'
#' @description Reads a dictionary file written by \code{catboost.dictionary.save}
#' or Python's \code{Dictionary.save}.
#'
#' @param dictionary A catboost.Dictionary object to load into.
#'
#' Default value: Required argument
#' @param frequency_dict_path Input file path for the frequency-based
#' alphabet.
#'
#' Default value: Required argument
#' @param bpe_path Input file path for Bpe merge data; when supplied, the
#' loaded dictionary is treated as "Bpe".
#'
#' Default value: NULL
#' @return The (mutated) catboost.Dictionary, invisibly.
#' @export
catboost.dictionary.load <- function(dictionary, frequency_dict_path, bpe_path = NULL) {
  if (!inherits(dictionary, "catboost.Dictionary")) {
    stop("catboost.dictionary.load: 'dictionary' must be a catboost.Dictionary object.")
  }
  dictionary$handle <- .Call("CatBoostTextDictionaryLoad_R", frequency_dict_path, bpe_path)
  dictionary$dictionary_type <- if (is.null(bpe_path)) "FrequencyBased" else "Bpe"
  dictionary$initialized <- TRUE
  invisible(dictionary)
}
