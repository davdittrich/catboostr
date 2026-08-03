#' @import jsonlite
NULL

# ---------------------------------------------------------------------------
# catboost.Tokenizer / catboost.Dictionary
#
# Pure-R port of the Python `catboost.text_processing.Tokenizer` and
# `catboost.text_processing.Dictionary` classes.
#
# Underlying logic location: the Python classes are thin Cython wrappers
# (vendor/catboost/catboost/python-package/catboost/_text_processing.pxi)
# around vendor/catboost/library/cpp/text_processing/{tokenizer,dictionary}.
# catboostr's src/ does not currently link that library and does not expose
# any tokenizer/dictionary bridge functions (confirmed: zero matches for
# "tokeniz"/"dictionary" under src/). Reusing the vendor implementation would
# require adding new Rcpp glue and wiring it into the build; instead this
# file is a pure-R re-implementation, verified byte-for-byte against the
# pinned Python oracle for the *default* configuration:
#   - Tokenizer: SeparatorType == "ByDelimiter" (the library default).
#     "BySense" mode wraps a large NLP sentence/word-boundary tokenizer
#     (library/cpp/tokenizer) that is not reasonably reimplementable in pure
#     R; it is explicitly unsupported here (clear error, not a silent
#     approximation).
#   - Dictionary: TokenLevelType == "Word", GramOrder == 1,
#     dictionary_type == "FrequencyBased" (the library defaults). Letter
#     n-grams, multigrams (GramOrder > 1) and Bpe dictionaries are out of
#     scope for this ticket (pure text-preprocessing vocabulary building) and
#     are rejected with a clear error.
# ---------------------------------------------------------------------------

.catboost.tp.is_number <- function(token) {
    grepl("^[0-9]+$", token)
}

.catboost.tp.split_by_delimiter <- function(string, delimiter, split_by_set, skip_empty) {
    if (nchar(string) == 0 && skip_empty) {
        return(character(0))
    }
    if (split_by_set) {
        chars <- unique(strsplit(delimiter, "", fixed = TRUE)[[1]])
        pattern <- paste0("[", paste(gsub("([][{}()*+?.\\^$|])", "\\\\\\1", chars), collapse = ""), "]")
        tokens <- strsplit(string, pattern, perl = TRUE)[[1]]
        # strsplit() drops a trailing empty field that SplitBySet() keeps; restore it.
        if (grepl(pattern, substring(string, nchar(string)), perl = TRUE)) {
            tokens <- c(tokens, "")
        }
    } else {
        tokens <- strsplit(string, delimiter, fixed = TRUE)[[1]]
        if (endsWith(string, delimiter) && nchar(delimiter) > 0) {
            tokens <- c(tokens, "")
        }
    }
    if (length(tokens) == 0) {
        tokens <- ""
    }
    if (skip_empty) {
        tokens <- tokens[nzchar(tokens)]
    }
    tokens
}

#' @name catboost.Tokenizer
#' @title Create a text Tokenizer
#'
#' @description Splits a string into tokens ahead of \code{catboost.Dictionary}
#' fitting/application. This is a pure-R port of Python's
#' \code{catboost.text_processing.Tokenizer}. Only the default
#' \code{separator_type = "ByDelimiter"} mode is supported; \code{"BySense"}
#' (full NLP sentence/word tokenization) is not reimplemented and raises an
#' error.
#'
#' @param lowercasing Lowercase every token. Default value: FALSE
#' @param number_process_policy One of "Skip", "LeaveAsIs", "Replace".
#'
#' Default value: "LeaveAsIs"
#' @param number_token Replacement token used when \code{number_process_policy}
#' is "Replace".
#'
#' Default value: "\\U0001F522" (matches the vendor default)
#' @param separator_type Only "ByDelimiter" is supported.
#'
#' Default value: "ByDelimiter"
#' @param delimiter Delimiter string used to split tokens.
#'
#' Default value: " "
#' @param split_by_set If TRUE, every individual character in \code{delimiter}
#' is treated as its own delimiter.
#'
#' Default value: FALSE
#' @param skip_empty Skip empty tokens produced by adjacent delimiters.
#'
#' Default value: TRUE
#' @return catboost.Tokenizer
#' @export
catboost.Tokenizer <- function(lowercasing = NULL,
                                number_process_policy = NULL,
                                number_token = NULL,
                                separator_type = NULL,
                                delimiter = NULL,
                                split_by_set = NULL,
                                skip_empty = NULL) {
    if (!is.null(separator_type) && separator_type != "ByDelimiter") {
        stop("catboost.Tokenizer: separator_type = '", separator_type, "' is not supported; ",
             "only 'ByDelimiter' (the library default) is implemented in the R port.")
    }
    number_process_policy <- if (is.null(number_process_policy)) "LeaveAsIs" else number_process_policy
    if (!(number_process_policy %in% c("Skip", "LeaveAsIs", "Replace"))) {
        stop("catboost.Tokenizer: unsupported number_process_policy '", number_process_policy, "'.")
    }

    structure(
        list(
            lowercasing = if (is.null(lowercasing)) FALSE else lowercasing,
            number_process_policy = number_process_policy,
            number_token = if (is.null(number_token)) "\U0001F522" else number_token,
            delimiter = if (is.null(delimiter)) " " else delimiter,
            split_by_set = if (is.null(split_by_set)) FALSE else split_by_set,
            skip_empty = if (is.null(skip_empty)) TRUE else skip_empty
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
#' @param types If TRUE, also return the token type ("Unknown" for every
#' token in "ByDelimiter" mode, matching the vendor implementation).
#'
#' Default value: FALSE
#' @return A character vector of tokens, or (if \code{types = TRUE}) a
#' data.frame with \code{token} and \code{type} columns.
#' @export
catboost.tokenizer.tokenize <- function(tokenizer, string, types = FALSE) {
    if (!inherits(tokenizer, "catboost.Tokenizer")) {
        stop("catboost.tokenizer.tokenize: 'tokenizer' must be a catboost.Tokenizer object.")
    }
    tokens <- .catboost.tp.split_by_delimiter(
        string, tokenizer$delimiter, tokenizer$split_by_set, tokenizer$skip_empty
    )

    if (tokenizer$lowercasing) {
        tokens <- tolower(tokens)
    }

    if (tokenizer$number_process_policy == "Replace") {
        is_num <- .catboost.tp.is_number(tokens)
        tokens[is_num] <- tokenizer$number_token
    } else if (tokenizer$number_process_policy == "Skip") {
        tokens <- tokens[!.catboost.tp.is_number(tokens)]
    }

    if (types) {
        return(data.frame(token = tokens, type = rep("Unknown", length(tokens)), stringsAsFactors = FALSE))
    }
    tokens
}


.catboost.tp.tokenize_line <- function(line, tokenizer) {
    # Mirrors _text_processing.pxi: a bare string is tokenized only when a
    # tokenizer is supplied; otherwise the whole string is treated as a
    # single token. An already-tokenized vector/list of tokens is passed
    # through unchanged, ignoring any tokenizer.
    if (is.character(line) && length(line) == 1) {
        if (!is.null(tokenizer)) {
            catboost.tokenizer.tokenize(tokenizer, line)
        } else {
            line
        }
    } else {
        as.character(line)
    }
}

.catboost.tp.default_dict_options <- function() {
    list(
        token_level_type = "Word",
        gram_order = 1L,
        start_token_id = 0L,
        end_of_word_policy = "Insert",
        end_of_sentence_policy = "Skip",
        occurence_lower_bound = 50L,
        max_dictionary_size = -1L,
        dictionary_type = "FrequencyBased"
    )
}

#' @name catboost.Dictionary
#' @title Create a text Dictionary
#'
#' @description Builds a token/id vocabulary from text data, mirroring the
#' default (Word-level, unigram, frequency-based) configuration of Python's
#' \code{catboost.text_processing.Dictionary}. Use \code{catboost.dictionary.fit}
#' to train it and \code{catboost.dictionary.apply} to convert text to token
#' ids. Letter-level tokens, multigrams (\code{gram_order > 1}) and
#' \code{dictionary_type = "Bpe"} are out of scope and rejected with a clear
#' error.
#'
#' @param token_level_type Only "Word" is supported.
#'
#' Default value: "Word"
#' @param gram_order Only 1 is supported.
#'
#' Default value: 1
#' @param start_token_id Initial shift for assigned token identifiers.
#'
#' Default value: 0
#' @param end_of_word_policy Unused for Word-level dictionaries; accepted for
#' interface parity.
#'
#' Default value: "Insert"
#' @param end_of_sentence_policy "Skip" or "Insert"; whether
#' \code{catboost.dictionary.apply} appends the end-of-sentence token id
#' after each tokenized line.
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
#' @param dictionary_type Only "FrequencyBased" is supported.
#'
#' Default value: "FrequencyBased"
#' @return catboost.Dictionary (a mutable object; \code{fit}/\code{load}
#' update it in place).
#' @export
catboost.Dictionary <- function(token_level_type = NULL,
                                 gram_order = NULL,
                                 start_token_id = NULL,
                                 end_of_word_policy = NULL,
                                 end_of_sentence_policy = NULL,
                                 occurence_lower_bound = NULL,
                                 max_dictionary_size = NULL,
                                 dictionary_type = NULL) {
    defaults <- .catboost.tp.default_dict_options()
    token_level_type <- if (is.null(token_level_type)) defaults$token_level_type else token_level_type
    gram_order <- if (is.null(gram_order)) defaults$gram_order else as.integer(gram_order)
    dictionary_type <- if (is.null(dictionary_type)) defaults$dictionary_type else dictionary_type

    if (token_level_type != "Word") {
        stop("catboost.Dictionary: token_level_type = '", token_level_type, "' is not supported; ",
             "only 'Word' (the library default) is implemented in the R port.")
    }
    if (gram_order != 1L) {
        stop("catboost.Dictionary: gram_order = ", gram_order, " is not supported; ",
             "only 1 (unigram) is implemented in the R port.")
    }
    if (dictionary_type != "FrequencyBased") {
        stop("catboost.Dictionary: dictionary_type = '", dictionary_type, "' is not supported; ",
             "only 'FrequencyBased' is implemented in the R port.")
    }
    end_of_sentence_policy <- if (is.null(end_of_sentence_policy)) defaults$end_of_sentence_policy else end_of_sentence_policy
    if (!(end_of_sentence_policy %in% c("Skip", "Insert"))) {
        stop("catboost.Dictionary: unsupported end_of_sentence_policy '", end_of_sentence_policy, "'.")
    }

    env <- new.env(parent = emptyenv())
    env$token_level_type <- token_level_type
    env$gram_order <- gram_order
    env$start_token_id <- if (is.null(start_token_id)) defaults$start_token_id else as.integer(start_token_id)
    env$end_of_word_policy <- if (is.null(end_of_word_policy)) defaults$end_of_word_policy else end_of_word_policy
    env$end_of_sentence_policy <- end_of_sentence_policy
    env$occurence_lower_bound <- if (is.null(occurence_lower_bound)) defaults$occurence_lower_bound else as.integer(occurence_lower_bound)
    env$max_dictionary_size <- if (is.null(max_dictionary_size)) defaults$max_dictionary_size else as.integer(max_dictionary_size)
    env$dictionary_type <- dictionary_type

    env$initialized <- FALSE
    env$token_to_id <- NULL # named integer vector: names = tokens, values = ids
    env$id_to_token <- NULL # character vector indexed by (id - start_token_id + 1)
    env$id_to_count <- NULL # integer vector, same indexing as id_to_token
    env$unknown_token_id <- NA_integer_
    env$end_of_sentence_token_id <- NA_integer_

    structure(env, class = "catboost.Dictionary")
}

.catboost.dictionary.check_initialized <- function(dictionary) {
    if (!isTRUE(dictionary$initialized)) {
        stop("catboost.Dictionary should be initialized (call catboost.dictionary.fit or ",
             "catboost.dictionary.load first).")
    }
}

#' @name catboost.dictionary.fit
#' @title Train a Dictionary
#'
#' @description Build the token/id vocabulary from text data.
#'
#' @param dictionary A catboost.Dictionary object; updated in place.
#'
#' Default value: Required argument
#' @param data A character vector (one document per element; tokenized with
#' \code{tokenizer} if given, otherwise each element is treated as a single
#' token) or a list of character vectors (already-tokenized documents).
#'
#' Default value: Required argument
#' @param tokenizer An optional catboost.Tokenizer used to split each
#' element of a character-vector \code{data}.
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

    counts <- new.env(hash = TRUE, parent = emptyenv())
    for (line in data) {
        tokens <- .catboost.tp.tokenize_line(line, tokenizer)
        for (tok in tokens) {
            counts[[tok]] <- if (is.null(counts[[tok]])) 1L else counts[[tok]] + 1L
        }
    }

    tokens <- ls(counts, sorted = FALSE)
    token_counts <- vapply(tokens, function(t) counts[[t]], integer(1))

    keep <- token_counts >= dictionary$occurence_lower_bound
    tokens <- tokens[keep]
    token_counts <- token_counts[keep]

    if (length(tokens) > 0) {
        ord <- order(token_counts, tokens, decreasing = c(TRUE, FALSE), method = "radix")
    } else {
        ord <- integer(0)
    }

    max_size <- if (dictionary$max_dictionary_size == -1L) length(ord) else min(length(ord), dictionary$max_dictionary_size)
    ord <- ord[seq_len(max_size)]

    id_to_token <- tokens[ord]
    id_to_count <- token_counts[ord]
    ids <- dictionary$start_token_id + seq_along(id_to_token) - 1L
    token_to_id <- stats::setNames(ids, id_to_token)

    dictionary$token_to_id <- token_to_id
    dictionary$id_to_token <- id_to_token
    dictionary$id_to_count <- id_to_count
    dictionary$unknown_token_id <- dictionary$start_token_id + length(id_to_token)
    dictionary$end_of_sentence_token_id <- dictionary$unknown_token_id + 1L
    dictionary$initialized <- TRUE

    invisible(dictionary)
}

#' @name catboost.dictionary.apply
#' @title Apply a Dictionary to text
#'
#' @description Convert text into token ids using a fitted
#' \code{catboost.Dictionary}.
#'
#' @param dictionary A fitted catboost.Dictionary object.
#'
#' Default value: Required argument
#' @param data A single string, a character vector (one document per
#' element), or a list of character vectors (already-tokenized documents).
#'
#' Default value: Required argument
#' @param tokenizer An optional catboost.Tokenizer used to split each raw
#' string in \code{data}.
#'
#' Default value: NULL
#' @param unknown_token_policy "Skip" (drop unknown tokens) or "Insert"
#' (emit \code{unknown_token_id}).
#'
#' Default value: "Skip"
#' @return An integer vector of token ids if \code{data} was a single
#' string, otherwise a list of integer vectors (one per document).
#' @export
catboost.dictionary.apply <- function(dictionary, data, tokenizer = NULL, unknown_token_policy = NULL) {
    .catboost.dictionary.check_initialized(dictionary)
    unknown_token_policy <- if (is.null(unknown_token_policy)) "Skip" else unknown_token_policy
    if (!(unknown_token_policy %in% c("Skip", "Insert"))) {
        stop("catboost.dictionary.apply: unsupported unknown_token_policy '", unknown_token_policy, "'.")
    }

    need_to_extract <- is.character(data) && length(data) == 1
    lines <- if (is.character(data)) as.list(data) else data
    if (!is.list(lines)) {
        stop("catboost.dictionary.apply: 'data' must be a string, a character vector, or a list of character vectors.")
    }

    apply_one <- function(line) {
        tokens <- .catboost.tp.tokenize_line(line, tokenizer)
        ids <- unname(dictionary$token_to_id[tokens])
        if (unknown_token_policy == "Insert") {
            ids[is.na(ids)] <- dictionary$unknown_token_id
        } else {
            ids <- ids[!is.na(ids)]
        }
        if (dictionary$end_of_sentence_policy == "Insert") {
            ids <- c(ids, dictionary$end_of_sentence_token_id)
        }
        as.integer(ids)
    }

    result <- lapply(lines, apply_one)
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
#' @return The number of tokens in the dictionary.
#' @export
catboost.dictionary.size <- function(dictionary) {
    .catboost.dictionary.check_initialized(dictionary)
    length(dictionary$id_to_token)
}

#' @name catboost.dictionary.get_token
#' @title Get a single token by id
#' @param dictionary A fitted catboost.Dictionary object.
#'
#' Default value: Required argument
#' @param token_id A token id.
#'
#' Default value: Required argument
#' @return The token string.
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
    vapply(token_ids, function(id) {
        if (id == dictionary$end_of_sentence_token_id) {
            return("_EOS_")
        }
        if (id == dictionary$unknown_token_id) {
            return("_UNK_")
        }
        idx <- id - dictionary$start_token_id + 1L
        if (idx < 1 || idx > length(dictionary$id_to_token)) {
            stop("catboost.dictionary.get_tokens: invalid token_id ", id, ".")
        }
        dictionary$id_to_token[idx]
    }, character(1))
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
    n <- min(top_size, length(dictionary$id_to_token))
    if (n <= 0) {
        return(character(0))
    }
    dictionary$id_to_token[seq_len(n)]
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
    dictionary$unknown_token_id
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
    dictionary$end_of_sentence_token_id
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
    dictionary$end_of_sentence_token_id + 1L
}

#' @name catboost.dictionary.save
#' @title Save a Dictionary to a file
#'
#' @description Writes the vendor "id_count_token" text format (a JSON
#' options header line, a token-count line, then one "id\\tcount\\ttoken"
#' line per token), which the pinned Python oracle's
#' \code{Dictionary.load} can also read.
#'
#' @param dictionary A fitted catboost.Dictionary object.
#'
#' Default value: Required argument
#' @param frequency_dict_path Output file path.
#'
#' Default value: Required argument
#' @return The catboost.Dictionary, invisibly.
#' @export
catboost.dictionary.save <- function(dictionary, frequency_dict_path) {
    .catboost.dictionary.check_initialized(dictionary)
    options_json <- jsonlite::toJSON(
        list(
            token_level_type = dictionary$token_level_type,
            gram_order = as.character(dictionary$gram_order),
            skip_step = "0",
            start_token_id = as.character(dictionary$start_token_id),
            end_of_word_token_policy = dictionary$end_of_word_policy,
            dictionary_format = "id_count_token",
            end_of_sentence_token_policy = dictionary$end_of_sentence_policy
        ),
        auto_unbox = TRUE
    )

    con <- file(frequency_dict_path, open = "wb")
    on.exit(close(con))
    writeLines(as.character(options_json), con = con, sep = "\n")
    writeLines(as.character(length(dictionary$id_to_token)), con = con, sep = "\n")
    if (length(dictionary$id_to_token) > 0) {
        ids <- dictionary$start_token_id + seq_along(dictionary$id_to_token) - 1L
        lines <- paste(ids, dictionary$id_to_count, dictionary$id_to_token, sep = "\t")
        writeLines(lines, con = con, sep = "\n")
    }

    invisible(dictionary)
}

#' @name catboost.dictionary.load
#' @title Load a Dictionary from a file
#'
#' @description Reads the vendor "id_count_token" text format written by
#' \code{catboost.dictionary.save} or by the pinned Python oracle's
#' \code{Dictionary.save}.
#'
#' @param dictionary A catboost.Dictionary object; updated in place.
#'
#' Default value: Required argument
#' @param frequency_dict_path Input file path.
#'
#' Default value: Required argument
#' @return The (mutated) catboost.Dictionary, invisibly.
#' @export
catboost.dictionary.load <- function(dictionary, frequency_dict_path) {
    if (!inherits(dictionary, "catboost.Dictionary")) {
        stop("catboost.dictionary.load: 'dictionary' must be a catboost.Dictionary object.")
    }
    lines <- readLines(frequency_dict_path, warn = FALSE)
    if (length(lines) < 2) {
        stop("catboost.dictionary.load: '", frequency_dict_path, "' is not a valid dictionary file.")
    }
    header <- jsonlite::fromJSON(lines[1])
    if (is.null(header$dictionary_format) || header$dictionary_format != "id_count_token") {
        stop("catboost.dictionary.load: only the 'id_count_token' dictionary format is supported by the R port.")
    }

    dict_size <- as.integer(lines[2])
    id_to_token <- character(dict_size)
    id_to_count <- integer(dict_size)
    ids <- integer(dict_size)
    if (dict_size > 0) {
        for (i in seq_len(dict_size)) {
            parts <- strsplit(lines[2 + i], "\t", fixed = TRUE)[[1]]
            ids[i] <- as.integer(parts[1])
            id_to_count[i] <- if (length(parts) >= 2 && nzchar(parts[2])) as.integer(parts[2]) else NA_integer_
            id_to_token[i] <- if (length(parts) > 3) paste(parts[3:length(parts)], collapse = "\t") else parts[3]
        }
    }
    ord <- order(ids)
    ids <- ids[ord]
    id_to_token <- id_to_token[ord]
    id_to_count <- id_to_count[ord]

    start_token_id <- if (!is.null(header$start_token_id)) as.integer(header$start_token_id) else 0L

    dictionary$token_level_type <- if (!is.null(header$token_level_type)) header$token_level_type else "Word"
    dictionary$gram_order <- if (!is.null(header$gram_order)) as.integer(header$gram_order) else 1L
    dictionary$start_token_id <- start_token_id
    dictionary$end_of_word_policy <- if (!is.null(header$end_of_word_token_policy)) header$end_of_word_token_policy else "Insert"
    dictionary$end_of_sentence_policy <- if (!is.null(header$end_of_sentence_token_policy)) header$end_of_sentence_token_policy else "Skip"
    dictionary$dictionary_type <- "FrequencyBased"

    dictionary$id_to_token <- id_to_token
    dictionary$id_to_count <- id_to_count
    dictionary$token_to_id <- stats::setNames(ids, id_to_token)
    dictionary$unknown_token_id <- start_token_id + dict_size
    dictionary$end_of_sentence_token_id <- dictionary$unknown_token_id + 1L
    dictionary$initialized <- TRUE

    invisible(dictionary)
}
