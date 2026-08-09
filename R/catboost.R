#' @import jsonlite
#' @importFrom stats predict
#' @importFrom utils head
#' @importFrom utils tail
#' @importFrom utils write.table
#' @importFrom utils download.file
#' @useDynLib libcatboostr, .registration = TRUE
NULL


#' @name catboost.load_pool
#' @title Create a dataset
#'
#' @description Create a dataset from the given file, matrix or data.frame.
#'
#' @param data A file path, matrix, sparse matrix (any \code{Matrix} package
#' \code{sparseMatrix}, e.g. \code{dgCMatrix}) or data.frame with features.
#' A sparse matrix is densified before being handed to the native Pool builder, so the resulting
#' Pool is identical to the one built from \code{as.matrix(data)} (stored zeros are real zeros).
#' Caveat: because the sparse layout is not preserved, training on such a Pool reproduces Python's
#' \emph{dense} Pool, not its native sparse one. CatBoost breaks ties between equal-scoring split
#' candidates differently in its sparse and dense column layouts, so on degenerate/tie-heavy data
#' the two can differ (a delta of 0.0298 was measured inside Python itself between its own dense
#' and sparse Pools).
#' The following column types are supported:
#' \itemize{
#'     \item double
#'     \item factor.
#'     It is assumed that categorical features are given in this type of columns.
#'     A standard CatBoost processing procedure is applied to this type of columns:
#'     \describe{
#'         \item{1.}{The values are converted to strings.}
#'         \item{2.}{The ConvertCatFeatureToFloat function is applied to the resulting string.}
#'     }
#' }
#'
#' Default value: Required argument
#' @param label The label vector or label matrix.
#' A plain (non-factor, non-character) integer label \emph{matrix} with more than one column
#' (e.g. a multi-target 0/1 matrix for \code{MultiLogloss}) is automatically read as a float
#' target, matching Python's \code{Pool}. A single-column integer label vector keeps its
#' existing integer-target behavior.
#' @param cat_features A vector of categorical features indices.
#' The indices are zero based and can differ from the given in the Column descriptions file.
#' If data parameter is data.frame don't use cat_features, categorical features are determined automatically
#' from data.frame column types.
#' @param column_description The path to the input file that contains the column descriptions.
#' @param embedding_features A named list of numeric matrices, one per embedding feature; each matrix
#' has one row per object and one column per embedding dimension. R matrix cells cannot hold vectors,
#' so embedding features are supplied beside \code{data} instead of inside it (Python's
#' \code{Pool(data, embedding_features = [...])} indexes columns of a data frame holding arrays).
#' The embedding features are appended after the columns of \code{data}, so their flat feature
#' indices are \code{ncol(data)}, \code{ncol(data) + 1}, ...; list names become their feature names.
#' Caveat: the \code{embedding_processing} training parameter (given to \code{catboost.train}, not
#' here) defaults to \code{list(default = list("LDA", "KNN"))}. The LDA calcer's output is not
#' bit-reproducible against the Python package: LDA solves a float32 symmetric eigenproblem whose
#' eigenvector signs and near-degenerate eigenvalue ordering depend on the LAPACK/BLAS build, so
#' this package and the Python wheel legitimately disagree. The KNN calcer agrees exactly. For
#' parity-sensitive use, pass \code{embedding_processing = list(default = list("KNN"))}.
#' @param pairs A file path, matrix or data.frame that contains the pairs descriptions. The shape should be Nx2, where N is the pairs' count.
#' The first element of pair is the index of winner document in training set. The second element of pair is the index of loser document in training set.
#' @param delimiter Delimiter character to use to separate features in a file.
#' @param has_header Read column names from first line, if this parameter is set to True.
#' @param weight The weights of the objects.
#' @param group_id The group ids of the objects.
#' @param group_weight The group weight of the objects.
#' @param subgroup_id The subgroup ids of the objects.
#' @param pairs_weight The weights of the pairs.
#' @param baseline Vector of initial (raw) values of the objective function.
#' Used in the calculation of final values of trees.
#' @param feature_names A list of names for each feature in the dataset.
#' @param thread_count The number of threads to use while reading the data. Optimizes reading time. This parameter doesn't affect results.
#' @param graph A file path, matrix or data.frame that contains the pairs of indices of objects for graph features.
#' The shape should be Nx2, where N is the pairs of indices count.
#' If -1, then the number of threads is set to the number of CPU cores.
#' @param timestamp A numeric vector of per-object timestamps, length equal to the number of
#' objects. Convenience wrapper around \code{\link{catboost.pool.set_timestamp}}: applied to the
#' constructed Pool before it is returned, equivalent to calling
#' \code{catboost.pool.set_timestamp(pool, timestamp)} afterward.
#' @param feature_tags A named list grouping features for
#' \code{\link{catboost.select_features}}'s \code{grouping = "ByTags"} mode
#' (mirrors Python's \code{Pool(feature_tags=...)}). Each element is itself a
#' list with a required \code{features} entry -- a vector of 0-based feature
#' indices or feature names (requires \code{feature_names} to be supplied) --
#' and an optional \code{cost} (default \code{1.0}), e.g.
#' \code{list(group_a = list(features = c(0, 1)), group_b = list(features = c("f2", "f3"), cost = 2))}.
#' Only supported when \code{data} is a matrix, sparse matrix or
#' \code{catboost.FeaturesData}; \code{NULL} (the default) attaches no tags.
#' Loading a pool from a file or a data.frame does not support feature tags;
#' passing a non-NULL value in those cases raises an error.
#'
#' @examples
#' \dontrun{
#' # From file
#' pool_path <- system.file("extdata", "adult_train.1000", package = "catboostr")
#' cd_path <- system.file("extdata", "adult.cd", package = "catboostr")
#' pool <- catboost.load_pool(pool_path, column_description = cd_path)
#' print(pool)
#'
#' # From matrix
#' target <- 1
#' data_matrix <-matrix(runif(18), 6, 3)
#' pool <- catboost.load_pool(data_matrix[, -target], label = data_matrix[, target])
#' print(pool)
#'
#' # From data.frame
#' nonsense <- factor(c('A', 'B', 'C'))
#' data_frame <- data.frame(value = runif(10), category = nonsense[(1:10) %% 3 + 1])
#' label = (1:10) %% 2
#' pool <- catboost.load_pool(data_frame, label = label)
#' print(pool)
#' }
#' @return catboost.Pool
#' @export
catboost.load_pool <- function(data, label = NULL, cat_features = NULL, column_description = NULL,
                               pairs = NULL, delimiter = "\t", has_header = FALSE, weight = NULL,
                               group_id = NULL, group_weight = NULL, subgroup_id = NULL, pairs_weight = NULL,
                               baseline = NULL, feature_names = NULL, thread_count = -1, graph = NULL,
                               embedding_features = NULL, timestamp = NULL, feature_tags = NULL) {
    if (!is.null(pairs) && (is.character(data) != is.character(pairs))) {
        stop("Data and pairs should be the same types.")
    }
    if (!is.null(graph) && (is.character(graph) != is.character(graph))) {
        stop("Data and graph should be the same types.")
    }
    if (!is.null(embedding_features) && !is.matrix(data) && !inherits(data, "sparseMatrix")) {
        stop("parameter 'embedding_features' is only supported when 'data' is a matrix or a sparse matrix")
    }
    if (!is.numeric(timestamp) && !is.null(timestamp))
        stop("Unsupported timestamp type, expecting numeric, got: ", typeof(timestamp))

    if (is.character(data) && length(data) == 1) {
        for (arg in list("label", "cat_features", "weight", "group_id",
                         "group_weight", "subgroup_id", "pairs_weight",
                         "baseline", "timestamp")) {
            if (!is.null(get(arg))) {
                stop("parameter '", arg, "' should be NULL when the pool is read from file")
            }
        }
        if (!is.null(feature_tags)) {
            stop("parameter 'feature_tags' should be NULL when the pool is read from file")
        }
        pool <- catboost.from_file(data, column_description, pairs, delimiter, has_header, thread_count, FALSE, feature_names, graph_path = graph)
    } else if (is.matrix(data) || inherits(data, "sparseMatrix")) {
        pool <- catboost.from_matrix(data, label, cat_features, NULL, NULL, pairs, weight, group_id, group_weight, subgroup_id, pairs_weight,
                                     baseline, feature_names, graph, embedding_features_data = embedding_features, timestamp = timestamp,
                                     feature_tags = feature_tags)
    } else if (inherits(data, "catboost.FeaturesData")) {
        for (arg in list("cat_features", "feature_names")) {
            if (!is.null(get(arg))) {
                stop("parameter '", arg, "' should be NULL when 'data' parameter has catboost.FeaturesData type")
            }
        }
        pool <- catboost.from_matrix(data, label, NULL, NULL, NULL, pairs, weight, group_id, group_weight, subgroup_id, pairs_weight,
                                     baseline, NULL, graph, timestamp = timestamp, feature_tags = feature_tags)
    } else if (is.data.frame(data)) {
        for (arg in list("column_description")) {
            if (!is.null(get(arg))) {
                stop("Parameter '", arg, "' should be NULL when the pool is constructed from data.frame")
            }
        }
        if (!is.null(feature_tags)) {
            stop("parameter 'feature_tags' should be NULL when the pool is constructed from data.frame")
        }
        if (!is.null(get("cat_features"))) {
            cat("Parameter 'cat_features' is meaningless because column types are taken from data.frame.",
                "Please, convert categorical columns to factors manually.", sep = "\n")
        }
        pool <- catboost.from_data_frame(data, label, pairs, weight, group_id, group_weight, subgroup_id, pairs_weight,
                                         baseline, feature_names, graph)
        if (!is.null(timestamp)) {
            if (length(timestamp) != nrow(data))
                stop("Data has ", nrow(data), " rows, timestamp vector has ", length(timestamp), " rows.")
            catboost.pool.set_timestamp(pool, timestamp)
        }
    } else {
        stop("Unsupported data type, expecting string, matrix, sparse matrix, data.frame or catboost.FeaturesData, got: ", class(data))
    }
    return(pool)
}


catboost.from_file <- function(pool_path, cd_path = "", pairs_path = "", delimiter = "\t", has_header = FALSE,
                               thread_count = -1, verbose = FALSE, feature_names_path = "",
                               num_vector_delimiter = ';', graph_path = "") {
    if (missing(pool_path))
        stop("Need to specify pool path.")
    if (is.null(pairs_path))
        pairs_path <- ""
    if (is.null(graph_path))
        graph_path <- ""
    if (is.null(feature_names_path))
        feature_names_path <- ""
    if (!is.character(pool_path) || !is.character(cd_path) || !is.character(pairs_path) || !is.character(graph_path) || !is.character(feature_names_path))
        stop("Path must be a string.")

    pool_path <- path.expand(pool_path)
    cd_path <- path.expand(cd_path)

    pool <- .Call("CatBoostCreateFromFile_R", pool_path, cd_path, pairs_path, graph_path, feature_names_path, delimiter, num_vector_delimiter, has_header, thread_count, verbose)
    attributes(pool) <- list(.Dimnames = list(NULL, NULL), class = "catboost.Pool")
    return(pool)
}


catboost.from_matrix <- function(float_and_cat_features_data, label = NULL, cat_features_indices = NULL, text_features_data = NULL,
                                 text_features_indices = NULL, pairs = NULL, weight = NULL, group_id = NULL, group_weight = NULL,
                                 subgroup_id = NULL, pairs_weight = NULL, baseline = NULL, feature_names = NULL, graph = NULL,
                                 embedding_features_data = NULL, embedding_features_indices = NULL, timestamp = NULL,
                                 feature_tags = NULL) {
  if (inherits(float_and_cat_features_data, "sparseMatrix")) {
      # ponytail: densify; CatBoost's sparse column format is a memory optimisation only, stored
      # zeros are ordinary zero values, so the resulting Pool equals the dense one. Upgrade path:
      # feed the dgCMatrix i/p/x slots to the native visitor's TConstPolymorphicValuesSparseArray
      # overloads if densification ever becomes the memory bottleneck.
      float_and_cat_features_data <- as.matrix(float_and_cat_features_data)
  }
  if (inherits(float_and_cat_features_data, "catboost.FeaturesData")) {
      if (!is.null(cat_features_indices) || !is.null(text_features_data) || !is.null(text_features_indices) || !is.null(feature_names)) {
          stop("parameters 'cat_features_indices', 'text_features_data', 'text_features_indices' and 'feature_names' should be NULL",
               " when 'float_and_cat_features_data' has catboost.FeaturesData type")
      }
      features_data <- float_and_cat_features_data
      num_data <- features_data$num_feature_data
      cat_data <- features_data$cat_feature_data
      num_ncol <- catboost.features_data.get_num_feature_count(features_data)
      cat_ncol <- catboost.features_data.get_cat_feature_count(features_data)

      if (!is.null(cat_data)) {
          cat_data <- matrix(.Call("CatBoostHashStrings_R", as.character(cat_data)), nrow = nrow(cat_data), ncol = ncol(cat_data))
      }
      float_and_cat_features_data <- if (!is.null(num_data) && !is.null(cat_data)) {
          cbind(num_data, cat_data)
      } else if (!is.null(num_data)) {
          num_data
      } else {
          cat_data
      }
      cat_features_indices <- if (cat_ncol > 0) as.integer(seq.int(num_ncol, num_ncol + cat_ncol - 1)) else integer(0)
      feature_names <- as.list(catboost.features_data.get_feature_names(features_data))
  }

  if (!is.matrix(float_and_cat_features_data))
      stop("Unsupported data type, expecting matrix or catboost.FeaturesData, got: ", class(float_and_cat_features_data))

  float_and_cat_columns <- if (is.null(float_and_cat_features_data)) 0 else ncol(float_and_cat_features_data)
  text_columns <- if (is.null(text_features_data)) 0 else ncol(text_features_data)
  embedding_columns <- if (is.null(embedding_features_data)) 0 else length(embedding_features_data)
  data_columns <- float_and_cat_columns + text_columns + embedding_columns
  if (text_columns == 0 && float_and_cat_columns == 0)
      stop("Data has no columns")

  if (!is.null(embedding_features_data)) {
      if (!is.list(embedding_features_data))
          stop("Unsupported embedding_features_data type, expecting list of matrices, got: ", class(embedding_features_data))
      for (embedding_index in seq_along(embedding_features_data)) {
          embedding <- embedding_features_data[[embedding_index]]
          if (!is.matrix(embedding) || !is.double(embedding))
              stop("embedding_features_data[[", embedding_index, "]] must be a double matrix, got: ", class(embedding))
          if (nrow(embedding) != nrow(float_and_cat_features_data))
              stop("Data has ", nrow(float_and_cat_features_data), " rows, embedding_features_data[[",
                   embedding_index, "]] has ", nrow(embedding), " rows.")
      }
      if (is.null(embedding_features_indices)) {
          # Embeddings travel beside the matrix, so they occupy the trailing flat feature indices.
          embedding_features_indices <- as.integer(seq.int(data_columns - embedding_columns, data_columns - 1))
      }
      if (length(embedding_features_indices) != embedding_columns)
          stop("embedding_features_data has ", embedding_columns, " features, embedding_features_indices has ",
               length(embedding_features_indices), " entries.")
      embedding_features_indices <- as.integer(embedding_features_indices)
      if (is.null(feature_names) && !is.null(names(embedding_features_data)) && text_columns == 0) {
          base_names <- colnames(float_and_cat_features_data)
          if (is.null(base_names))
              base_names <- as.character(seq_len(float_and_cat_columns) - 1L)
          feature_names <- as.list(c(base_names, names(embedding_features_data)))
      }
  } else if (!is.null(embedding_features_indices)) {
      stop("embedding_features_indices was given without embedding_features_data")
  }

  if (is.character(label))
      label <- as.factor(label)

  if (is.factor(label)) {
      class_labels <- labels(label)
      # R starts integer labels from 1, we generally prefer to start from 0
      label <- as.integer(label) - 1L
  } else {
      class_labels <- NULL
  }

  if (!is.null(label) && !is.matrix(label))
      label <- as.matrix(label)
  # A plain (non-factor, non-character) integer label matrix with more than
  # one column is a multi-target numeric label (e.g. MultiLogloss/MultiRMSE),
  # never a class-label encoding -- those only ever produce a single column
  # via the is.factor() branch above, which sets class_labels and must keep
  # its Integer storage mode untouched. Coerce to double so C++ dispatches
  # ERawTargetType::Float here, matching Python's Pool(data, label=<int
  # ndarray>) target-type semantics (catboost-8z4.47).
  if (!is.null(label) && is.null(class_labels) && is.integer(label) && ncol(label) > 1L)
      storage.mode(label) <- "double"
  if (!is.double(label) && !is.integer(label) && !is.null(label))
      stop("Unsupported label type, expecting double or int, got: ", typeof(label))

  if (!is.null(label) && nrow(label) != nrow(float_and_cat_features_data))
      stop("Data has ", nrow(float_and_cat_features_data), " rows, label has ", nrow(label), " rows.")

  if (!all(cat_features_indices == as.integer(cat_features_indices)) && !is.null(cat_features_indices))
      stop("Unsupported cat_features_indices type, expecting integer, got: ", typeof(cat_features_indices))

  if (!is.null(text_features_data) && !is.matrix(text_features_data))
     stop("Unsupported text data type, expecting matrix, got: ", class(text_features_data))
  if (!all(text_features_indices == as.integer(text_features_indices)) && !is.null(text_features_indices))
     stop("Unsupported text_features_indices type, expecting integer, got: ", typeof(text_features_indices))

  if (!is.matrix(pairs) && !is.null(pairs))
      stop("Unsupported pairs class, expecting matrix, got: ", class(pairs))
  if (!is.null(pairs) && dim(pairs)[2] != 2)
      stop("Unsupported pairs dim, expecting 2 columns, got: ", dim(pairs)[2])
  if (!all(pairs == as.integer(pairs)) && !is.null(pairs))
      stop("Unsupported pair type, expecting integer, got: ", typeof(pairs))

  if (!is.matrix(graph) && !is.null(graph))
      stop("Unsupported graph class, expecting matrix, got: ", class(graph))
  if (!is.null(graph) && dim(graph)[2] != 2)
      stop("Unsupported graph dim, expecting 2 columns, got: ", dim(graph)[2])
  if (!all(graph == as.integer(graph)) && !is.null(graph))
      stop("Unsupported graph data elements type, expecting integer, got: ", typeof(graph))

  if (!is.double(weight) && !is.null(weight))
      stop("Unsupported weight type, expecting double, got: ", typeof(weight))
  if (length(weight) != nrow(float_and_cat_features_data) && !is.null(weight))
      stop("Data has ", nrow(float_and_cat_features_data), " rows, weight vector has ", length(weight), " rows.")

  if (!is.integer(group_id) && !is.null(group_id))
      stop("Unsupported group_id type, expecting int, got: ", typeof(group_id))
  if (length(group_id) != nrow(float_and_cat_features_data) && !is.null(group_id))
      stop("Data has ", nrow(float_and_cat_features_data), " rows, group_id vector has ", length(group_id), " rows.")

  if (!is.double(group_weight) && !is.null(group_weight))
      stop("Unsupported group_weight type, expecting double, got: ", typeof(group_weight))
  if (length(group_weight) != nrow(float_and_cat_features_data) && !is.null(group_weight))
      stop("Data has ", nrow(float_and_cat_features_data), " rows, group_weight vector has ", length(group_weight), " rows.")

  if (!is.integer(subgroup_id) && !is.null(subgroup_id))
      stop("Unsupported subgroup_id type, expecting int, got: ", typeof(subgroup_id))
  if (length(subgroup_id) != nrow(float_and_cat_features_data) && !is.null(subgroup_id))
      stop("Data has ", nrow(float_and_cat_features_data), " rows, subgroup_id vector has ", length(subgroup_id), " rows.")

  if (!is.double(pairs_weight) && !is.null(pairs_weight))
      stop("Unsupported pairs_weight type, expecting double, got: ", typeof(pairs_weight))
  if (length(pairs_weight) != nrow(pairs) && !is.null(pairs_weight) && !is.null(pairs))
      stop("Pairs has ", nrow(pairs), " rows, pairs_weight vector has ", length(pairs_weight), " rows.")

  if (!is.matrix(baseline) && !is.null(baseline))
      stop("Baseline should be matrix, got: ", class(baseline))
  if (!is.double(baseline) && !is.null(baseline))
      stop("Unsupported baseline type, expecting double, got: ", typeof(baseline))
  if (nrow(baseline) != nrow(float_and_cat_features_data) && !is.null(baseline))
      stop("Baseline must be matrix of size n_objects*n_classes. Data has ", nrow(float_and_cat_features_data), " objects, baseline has ", nrow(baseline), " rows.")

  if (!is.list(feature_names) && !is.null(feature_names))
      stop("Unsupported feature_names type, expecting list, got: ", typeof(feature_names))
  if (!is.null(feature_names) && (length(feature_names) != data_columns))
      stop("Data has ", data_columns, " columns, feature_names has ", length(feature_names), " columns.")

  # catboost-8z4.121: feature-tags plumbing for catboost.select_features's
  # grouping = "ByTags" (mirrors Python's Pool(feature_tags=...), a named
  # dict of {'features': [...], 'cost': <number>} -- core.py:1080-1101). Each
  # tag's features may be given as 0-based indices or (when feature_names is
  # supplied) feature name strings; resolved to plain 0-based integer indices
  # here so the native side (src/catboostr.cpp's GetFeatureTagsFromSEXP) only
  # has to trust well-formed input, same division of labor as this function's
  # other index arguments (cat/text/embedding_features_indices).
  if (!is.null(feature_tags)) {
      if (!is.list(feature_tags) || is.null(names(feature_tags)) ||
          any(names(feature_tags) == "") || anyDuplicated(names(feature_tags)))
          stop("feature_tags must be a named list with unique, non-empty names")
      tag_names <- names(feature_tags)
      feature_tags <- setNames(lapply(tag_names, function(tag_name) {
          entry <- feature_tags[[tag_name]]
          if (!is.list(entry) || is.null(entry$features))
              stop("feature_tags[['", tag_name, "']] must be a list with a 'features' element")
          features <- entry$features
          if (is.character(features)) {
              if (is.null(feature_names))
                  stop("feature_tags[['", tag_name, "']]$features was given as name(s), but no feature_names were supplied")
              idx <- match(features, as.character(feature_names))
              if (any(is.na(idx)))
                  stop("feature_tags[['", tag_name, "']]$features has name(s) not present in feature_names: ",
                       paste(features[is.na(idx)], collapse = ", "))
              features <- idx - 1L
          } else if (is.numeric(features)) {
              features <- as.integer(features)
          } else {
              stop("feature_tags[['", tag_name, "']]$features must be a numeric or character vector, got: ", typeof(features))
          }
          if (length(features) == 0 || anyNA(features) || any(features < 0L | features >= data_columns))
              stop("feature_tags[['", tag_name, "']]$features must be non-empty 0-based indices in [0, ", data_columns - 1, "]")
          cost <- if (is.null(entry$cost)) 1.0 else as.double(entry$cost)
          if (length(cost) != 1 || is.na(cost))
              stop("feature_tags[['", tag_name, "']]$cost must be a single numeric value")
          list(features = features, cost = cost)
      }), tag_names)
  }

  if (!is.numeric(timestamp) && !is.null(timestamp))
      stop("Unsupported timestamp type, expecting numeric, got: ", typeof(timestamp))
  if (length(timestamp) != nrow(float_and_cat_features_data) && !is.null(timestamp))
      stop("Data has ", nrow(float_and_cat_features_data), " rows, timestamp vector has ", length(timestamp), " rows.")

  if (float_and_cat_columns == 0)
      float_and_cat_features_data <- NULL
  if (text_columns == 0)
      text_features_data <- NULL
  pool <- .Call("CatBoostCreateFromMatrix_R",
                float_and_cat_features_data, label, cat_features_indices, text_features_data, text_features_indices, pairs, graph, weight,
                group_id, group_weight, subgroup_id, pairs_weight, baseline, feature_names, class_labels,
                embedding_features_data, embedding_features_indices, feature_tags)
  attributes(pool) <- list(.Dimnames = list(NULL, as.character(feature_names)), class = "catboost.Pool")
  if (!is.null(timestamp))
      catboost.pool.set_timestamp(pool, timestamp)
  return(pool)
}


catboost.from_data_frame <- function(data, label = NULL, pairs = NULL, weight = NULL, group_id = NULL, group_weight = NULL,
                                     subgroup_id = NULL, pairs_weight = NULL, baseline = NULL, feature_names = NULL, graph = NULL) {
    if (!is.data.frame(data)) {
        stop("Unsupported data type, expecting data.frame, got: ", class(data))
    }
    if (is.null(feature_names)) {
        feature_names <- as.list(colnames(data))
    }
    factor_columns <- vapply(data, is.factor, logical(1))
    num_columns <-
      vapply(data, is.double, logical(1)) |
      vapply(data, is.integer, logical(1)) |
      vapply(data, is.logical, logical(1))
    text_columns <- vapply(data, is.character, logical(1))
    bad_columns <- !(factor_columns | num_columns | text_columns)

    if (sum(bad_columns) > 0) {
        stop("Unsupported column type: ", paste(c(unique(vapply(data[, bad_columns], class, character(1)))), collapse = ", "))
    }

    text_features_data <- data[text_columns]
    text_features_indices <- c()
    for (column_index in which(text_columns)){
        text_features_indices <- c(text_features_indices, column_index - 1)
    }

    float_and_cat_features_data <- data
    cat_features_indices <- c()
    for (column_index in which(factor_columns)) {
        float_and_cat_features_data[, column_index] <- .Call("CatBoostHashStrings_R", as.character(float_and_cat_features_data[[column_index]]))
        cat_features_indices <- c(cat_features_indices, column_index - 1)
    }
    float_and_cat_features_data <- float_and_cat_features_data[!text_columns]

    if (!is.null(pairs)) {
        pairs <- as.matrix(pairs)
    }
    if (!is.null(graph)) {
        graph <- as.matrix(graph)
    }
    pool <- catboost.from_matrix(as.matrix(float_and_cat_features_data), label, cat_features_indices, as.matrix(text_features_data),
                                 text_features_indices, pairs, weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, feature_names, graph)
    return(pool)
}


.catboost.check_features_data_part <- function(part_name, feature_data, is_numeric_part, feature_names) {
    if (!is.null(feature_names) && is.null(feature_data)) {
        stop(part_name, "_feature_names specified with not specified ", part_name, "_feature_data")
    }
    if (!is.null(feature_data)) {
        if (!is.matrix(feature_data)) {
            stop("only matrix type is supported for ", part_name, "_feature_data")
        }
        if (is_numeric_part && !is.numeric(feature_data)) {
            stop(part_name, "_feature_data element type must be numeric, found ", typeof(feature_data), " instead")
        }
        if (!is_numeric_part && !is.character(feature_data)) {
            stop(part_name, "_feature_data element type must be character, found ", typeof(feature_data), " instead")
        }
        if (!is.null(feature_names) && ncol(feature_data) != length(feature_names)) {
            stop("number of features in ", part_name, "_feature_data (=", ncol(feature_data),
                ") is different from length(", part_name, "_feature_names) (=", length(feature_names), ")")
        }
    }
    if (is.null(feature_names)) {
        feature_names <- if (!is.null(feature_data)) as.character(rep("", ncol(feature_data))) else character(0)
    } else {
        feature_names <- as.character(feature_names)
    }
    feature_names
}


#' @name catboost.FeaturesData
#' @title Create a FeaturesData container
#'
#' @description Store features data in a form that can be passed directly to
#' \code{catboost.load_pool}/\code{catboost.from_matrix} as the \code{data}/
#' \code{float_and_cat_features_data} argument, as an alternative to a plain
#' matrix or data.frame. Numerical features are given as a numeric matrix,
#' categorical features are given separately as a character matrix, each with
#' optional column names.
#'
#' @param num_feature_data A numeric matrix of numerical feature values, or NULL.
#'
#' Default value: NULL
#' @param cat_feature_data A character matrix of categorical feature values, or NULL.
#'
#' Default value: NULL
#' @param num_feature_names A list/vector of names for the numerical features. Must
#' be NULL if num_feature_data is NULL. If not specified, empty strings are used.
#'
#' Default value: NULL
#' @param cat_feature_names A list/vector of names for the categorical features. Must
#' be NULL if cat_feature_data is NULL. If not specified, empty strings are used.
#'
#' Default value: NULL
#' @return catboost.FeaturesData
#' @export
catboost.FeaturesData <- function(num_feature_data = NULL, cat_feature_data = NULL,
                                  num_feature_names = NULL, cat_feature_names = NULL) {
    if (is.null(num_feature_data) && is.null(cat_feature_data)) {
        stop("at least one of num_feature_data, cat_feature_data params must be non-NULL")
    }

    num_feature_names <- .catboost.check_features_data_part("num", num_feature_data, TRUE, num_feature_names)
    cat_feature_names <- .catboost.check_features_data_part("cat", cat_feature_data, FALSE, cat_feature_names)

    all_feature_count <- (if (!is.null(num_feature_data)) ncol(num_feature_data) else 0) +
                         (if (!is.null(cat_feature_data)) ncol(cat_feature_data) else 0)
    if (all_feature_count == 0) {
        stop("both num_feature_data and cat_feature_data contain 0 features")
    }

    if (!is.null(num_feature_data) && !is.null(cat_feature_data) && nrow(num_feature_data) != nrow(cat_feature_data)) {
        stop("object_counts in num_feature_data (", nrow(num_feature_data), ") and in cat_feature_data (",
            nrow(cat_feature_data), ") are different")
    }

    structure(
        list(
            num_feature_data = num_feature_data,
            cat_feature_data = cat_feature_data,
            num_feature_names = num_feature_names,
            cat_feature_names = cat_feature_names
        ),
        class = "catboost.FeaturesData"
    )
}


#' @name catboost.features_data.get_object_count
#' @title Number of objects in a FeaturesData
#' @description Get the number of objects (rows) in a catboost.FeaturesData.
#' @param features_data A catboost.FeaturesData object.
#'
#' Default value: Required argument
#' @return The number of objects.
#' @export
catboost.features_data.get_object_count <- function(features_data) {
    if (!is.null(features_data$num_feature_data)) {
        return(nrow(features_data$num_feature_data))
    }
    return(nrow(features_data$cat_feature_data))
}


#' @name catboost.features_data.get_num_feature_count
#' @title Number of numerical features in a FeaturesData
#' @description Get the number of numerical features in a catboost.FeaturesData.
#' @param features_data A catboost.FeaturesData object.
#'
#' Default value: Required argument
#' @return The number of numerical features.
#' @export
catboost.features_data.get_num_feature_count <- function(features_data) {
    if (is.null(features_data$num_feature_data)) {
        return(0L)
    }
    return(ncol(features_data$num_feature_data))
}


#' @name catboost.features_data.get_cat_feature_count
#' @title Number of categorical features in a FeaturesData
#' @description Get the number of categorical features in a catboost.FeaturesData.
#' @param features_data A catboost.FeaturesData object.
#'
#' Default value: Required argument
#' @return The number of categorical features.
#' @export
catboost.features_data.get_cat_feature_count <- function(features_data) {
    if (is.null(features_data$cat_feature_data)) {
        return(0L)
    }
    return(ncol(features_data$cat_feature_data))
}


#' @name catboost.features_data.get_feature_count
#' @title Total number of features in a FeaturesData
#' @description Get the total number of features (numerical + categorical) in a
#' catboost.FeaturesData.
#' @param features_data A catboost.FeaturesData object.
#'
#' Default value: Required argument
#' @return The total number of features.
#' @export
catboost.features_data.get_feature_count <- function(features_data) {
    return(catboost.features_data.get_num_feature_count(features_data) +
          catboost.features_data.get_cat_feature_count(features_data))
}


#' @name catboost.features_data.get_feature_names
#' @title Get feature names from a FeaturesData
#' @description Get the names of the features of a catboost.FeaturesData:
#' numerical feature names followed by categorical feature names. Unnamed
#' features are empty strings.
#' @param features_data A catboost.FeaturesData object.
#'
#' Default value: Required argument
#' @return A character vector of feature names, length equal to
#' \code{catboost.features_data.get_feature_count(features_data)}.
#' @export
catboost.features_data.get_feature_names <- function(features_data) {
    return(c(features_data$num_feature_names, features_data$cat_feature_names))
}


#' @name catboost.save_pool
#' @title Save the dataset
#'
#' @description Save the dataset to the CatBoost format.
#'              Files with the following data are created:
#'              \itemize{
#'                  \item Dataset description
#'                  \item Column descriptions
#'              }
#'              Use the catboost.load_pool function to read the resulting files.
#'              These files can also be used in the
#'              \href{https://catboost.ai/docs/concepts/cli-installation.html}{Command-line version}
#'              and the \href{https://catboost.ai/docs/concepts/python-installation.html}{Python library}.
#'
#' @param data A data.frame with features.
#' The following column types are supported:
#'     \itemize{
#'     \item double
#'     \item factor.
#'     It is assumed that categorical features are given in this type of columns.
#'     A standard CatBoost processing procedure is applied to this type of columns:
#'     \describe{
#'         \item{1.}{The values are converted to strings.}
#'         \item{2.}{The ConvertCatFeatureToFloat function is applied to the resulting string.}
#'     }
#' }
#'
#' Default value: Required argument
#' @param label The label vector.
#' @param weight The weights of the label vector.
#' @param baseline Vector of initial (raw) values of the label function for the object.
#' Used in the calculation of final values of trees.
#' @param pool_path The path to the output file that contains the dataset description.
#' @param cd_path The path to the output file that contains the column descriptions.
#' @return Nothing. This method writes a dataset to disk.
#' @export
catboost.save_pool <- function(data, label = NULL, weight = NULL, baseline = NULL,
                               pool_path = "data.pool", cd_path = "cd.pool") {
    if (missing(pool_path) || missing(cd_path))
        stop("Need to specify pool_path and cd_path.")
    if (!is.character(pool_path) || !is.character(cd_path))
        stop("Path must be a string.")


    pool <- label
    if (!is.null(pool)) {
        column_description <- c("Label")
    }
    if (is.null(weight) == FALSE) {
        pool <- cbind(pool, weight)
        column_description <- c(column_description, "Weight")
    }
    if (is.null(baseline) == FALSE) {
        baseline <- matrix(baseline, nrow = nrow(data), byrow = TRUE)
        pool <- cbind(pool, baseline)
        column_description <- c(column_description, rep("Baseline", ncol(baseline)))
    }
    pool <- cbind(pool, data)
    column_description <- data.frame(index = seq(0, length(column_description) - 1), type = column_description)
    factors <- which(vapply(data, class, character(1)) == "factor")
    if (length(factors) != 0) {
        column_description <- rbind(column_description, data.frame(index = nrow(column_description) + factors - 1,
                                                                   type = rep("Categ", length(factors))))
    }
    pool_path <- path.expand(pool_path)
    cd_path <- path.expand(cd_path)
    write.table(pool, file = pool_path, sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
    write.table(column_description, file = cd_path, sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
}


#' @name dim.catboost.Pool
#' @title Dimensions of catboost.Pool
#' @description Get dimensions of a Pool.
#'
#' @param x The input dataset.
#'
#' Default value: Required argument
#' @return Returns a vector of row numbers and column numbers in an catboost.Pool.
#' @export
dim.catboost.Pool <- function(x) {
    return(c(.Call("CatBoostPoolNumRow_R", x), .Call("CatBoostPoolNumCol_R", x)))
}


#' @name dimnames.catboost.Pool
#' @title Dimension names of catboost.Pool
#'
#' @description Get dimension names of a Pool.
#'
#' Column names are read from the Pool's own (C++-side) feature layout, so
#' they stay in sync with \code{catboost.pool.get_feature_names()} after a
#' \code{catboost.pool.set_feature_names()} call, which mutates that layout in
#' place and cannot update an R-side attribute of the caller's object.
#' @param x The input dataset.
#'
#' Default value: Required argument
#' @return A list with the two elements. The second element contains the column
#' names, or \code{NULL} if the Pool has no feature names.
#' @export
dimnames.catboost.Pool <- function(x) {
    if (is.null.handle(x))
        stop("Pool object is invalid.")
    feature_names <- .Call("CatBoostPoolGetFeatureNames_R", x)
    if (length(feature_names) == 0 || all(feature_names == ""))
        feature_names <- NULL
    return(list(NULL, feature_names))
}


#' @name head.catboost.Pool
#' @title Head of catboost.Pool
#'
#' @description Return a list with the first n objects of the dataset.
#'
#'              Each line of this list contains the following information for each object:
#'              \itemize{
#'                  \item The label value.
#'                  \item The weight value.
#'                  \item The feature values.
#'              }
#' @param x The input dataset.
#'
#' Default value: Required argument
#' @param n The quantity of the first objects in the dataset to be returned.
#'
#' Default value: 10
#' @param ... not currently used
#' @return A matrix containing the first \code{n} objects of the dataset.
#' @export
head.catboost.Pool <- function(x, n = 10, ...) {
    if (is.null.handle(x))
        stop("Pool object is invalid.")
    if (n < 0) {
        n <- max(0, dim(x)[1] + n)
    } else {
        n <- min(n, dim(x)[1])
    }
    result <- .Call("CatBoostPoolSlice_R", x, n, 0)
    result <- matrix(unlist(result), nrow = n, byrow = TRUE)
    return(result)
}

#' @name tail.catboost.Pool
#' @title Tail of catboost.Pool
#'
#' @description Return a list with the last n objects of the dataset.
#'
#'              Each line of this list contains the following information for each object:
#'              \itemize{
#'                  \item The target value.
#'                  \item The weight value.
#'                  \item The feature values.
#'              }
#' @param x The input dataset.
#'
#' Default value: Required argument
#' @param n The quantity of the last objects in the dataset to be returned.
#'
#' Default value: 10
#' @param ... not currently used
#' @return A matrix containing the last \code{n} objects of the dataset.
#' @export
tail.catboost.Pool <- function(x, n = 10, ...) {
    if (is.null.handle(x))
        stop("Pool object is invalid.")
    if (n < 0) {
        n <- max(0, dim(x)[1] + n)
    } else {
        n <- min(n, dim(x)[1])
    }
    result <- .Call("CatBoostPoolSlice_R", x, n, dim(x)[1] - n)
    result <- matrix(unlist(result), nrow = n, byrow = TRUE)
    return(result)
}

#' @name print.catboost.Pool
#' @title Print catboost.Pool
#'
#' @description Print dimensions of catboost.Pool.
#'
#' @param x a catboost.Pool object
#'
#' Default value: Required argument
#' @param ... not currently used
#' @return Nothing. This method prints pool dimensions.
#' @export
print.catboost.Pool <- function(x, ...) {
    if (is.null.handle(x))
        cat("Warning: pool object is invalid.")
    cat("catboost.Pool\n", nrow(x), " rows, ", ncol(x), " columns", sep = "")
}


# P3.1: canonicalize an id vector (group_id/subgroup_id) to the decimal-
# string tokens CalcGroupIdFor()/CalcSubgroupIdFor() hash, matching Python's
# get_id_object_bytes_string_representation(): integral values format as
# plain decimal (no ".0"), character values pass through unchanged. Floats
# with a fractional part are rejected, same as the Python method.
id.tokens.from.vector <- function(ids, arg_name) {
    if (is.character(ids)) {
        return(ids)
    }
    if (is.numeric(ids)) {
        if (any(ids != floor(ids))) {
            stop(arg_name, " must be integral or character valued.")
        }
        return(sprintf("%.0f", ids))
    }
    stop("Unsupported ", arg_name, " type, expecting character or numeric, got: ", typeof(ids))
}


#' @name catboost.pool.has_label
#' @title Has the Pool got label data
#' @description Check whether the Pool has label (target) data.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @return \code{TRUE} if the Pool has label data, \code{FALSE} otherwise.
#' @export
catboost.pool.has_label <- function(pool) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    return(.Call("CatBoostPoolHasLabel_R", pool))
}


#' @name catboost.pool.get_label
#' @title Get labels from a Pool
#' @description Get the label (target) data of a Pool.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @return A vector of labels if the target is one-dimensional, a
#' (rows x targets) matrix otherwise. \code{numeric(0)} if the Pool has no
#' label data.
#' @export
catboost.pool.get_label <- function(pool) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    return(.Call("CatBoostPoolGetLabel_R", pool))
}


#' @name catboost.pool.get_weight
#' @title Get weights from a Pool
#' @description Get the per-object weight of a Pool. Objects with no weight
#' set default to weight 1.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @return A numeric vector of per-object weights.
#' @export
catboost.pool.get_weight <- function(pool) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    return(.Call("CatBoostPoolGetWeight_R", pool))
}


#' @name catboost.pool.set_weight
#' @title Set weights on a Pool
#' @description Set the per-object weight of a Pool.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @param weight A numeric vector of per-object weights, length equal to
#' \code{nrow(pool)}.
#'
#' Default value: Required argument
#' @return Nothing. Mutates \code{pool} in place.
#' @export
catboost.pool.set_weight <- function(pool, weight) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    invisible(.Call("CatBoostPoolSetWeight_R", pool, as.double(weight)))
}


#' @name catboost.pool.get_baseline
#' @title Get baseline from a Pool
#' @description Get the baseline data of a Pool.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @return A (rows x baseline_count) numeric matrix. \code{baseline_count}
#' is 0 if the Pool has no baseline data.
#' @export
catboost.pool.get_baseline <- function(pool) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    return(.Call("CatBoostPoolGetBaseline_R", pool))
}


#' @name catboost.pool.set_baseline
#' @title Set baseline on a Pool
#' @description Set the baseline data of a Pool.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @param baseline A (rows x baseline_count) numeric matrix, \code{rows}
#' equal to \code{nrow(pool)}.
#'
#' Default value: Required argument
#' @return Nothing. Mutates \code{pool} in place.
#' @export
catboost.pool.set_baseline <- function(pool, baseline) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    if (!is.matrix(baseline))
        stop("baseline must be a matrix.")
    invisible(.Call("CatBoostPoolSetBaseline_R", pool, matrix(as.double(baseline), nrow = nrow(baseline))))
}


#' @name catboost.pool.get_group_id_hash
#' @title Get group id hashes from a Pool
#' @description Get the hashes generated from a Pool's group ids.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @return A character vector of decimal-formatted 64-bit hash values (one
#' per object), or \code{NULL} if the Pool has no group ids. Returned as
#' character rather than numeric because R's double cannot represent the
#' full 64-bit hash range exactly.
#' @export
catboost.pool.get_group_id_hash <- function(pool) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    return(.Call("CatBoostPoolGetGroupIdHash_R", pool))
}


#' @name catboost.pool.set_group_id
#' @title Set group ids on a Pool
#' @description Set the group ids of a Pool. Each id is hashed the same way
#' Python's \code{Pool.set_group_id} hashes it, so the resulting group id
#' hashes (see \code{\link{catboost.pool.get_group_id_hash}}) match the
#' Python oracle for the same input values.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @param group_id A character or integral-numeric vector, length equal to
#' \code{nrow(pool)}.
#'
#' Default value: Required argument
#' @return Nothing. Mutates \code{pool} in place.
#' @export
catboost.pool.set_group_id <- function(pool, group_id) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    tokens <- id.tokens.from.vector(group_id, "group_id")
    invisible(.Call("CatBoostPoolSetGroupId_R", pool, tokens))
}


#' @name catboost.pool.set_group_weight
#' @title Set group weights on a Pool
#' @description Set the per-object group weight of a Pool (weights must be
#' equal within each group).
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @param group_weight A numeric vector of per-object group weights, length
#' equal to \code{nrow(pool)}.
#'
#' Default value: Required argument
#' @return Nothing. Mutates \code{pool} in place.
#' @export
catboost.pool.set_group_weight <- function(pool, group_weight) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    invisible(.Call("CatBoostPoolSetGroupWeight_R", pool, as.double(group_weight)))
}


#' @name catboost.pool.set_subgroup_id
#' @title Set subgroup ids on a Pool
#' @description Set the subgroup ids of a Pool. Each id is hashed the same
#' way Python's \code{Pool.set_subgroup_id} hashes it.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @param subgroup_id A character or integral-numeric vector, length equal
#' to \code{nrow(pool)}.
#'
#' Default value: Required argument
#' @return Nothing. Mutates \code{pool} in place.
#' @export
catboost.pool.set_subgroup_id <- function(pool, subgroup_id) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    tokens <- id.tokens.from.vector(subgroup_id, "subgroup_id")
    invisible(.Call("CatBoostPoolSetSubgroupId_R", pool, tokens))
}


#' @name catboost.pool.set_pairs
#' @title Set pairs on a Pool
#' @description Set the pairwise comparison data of a Pool.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @param pairs An (N x 2) or (N x 3) numeric matrix of
#' \code{(winner_id, loser_id[, weight])} rows. \code{winner_id}/\code{loser_id}
#' are 0-indexed object row numbers, matching the \code{pairs} argument of
#' \code{\link{catboost.from_matrix}}. \code{weight} defaults to 1.0 when the
#' third column is omitted.
#'
#' Default value: Required argument
#' @return Nothing. Mutates \code{pool} in place.
#' @export
catboost.pool.set_pairs <- function(pool, pairs) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    if (!is.matrix(pairs) || !(ncol(pairs) %in% c(2, 3)))
        stop("pairs must be an (N x 2) or (N x 3) matrix.")
    invisible(.Call("CatBoostPoolSetPairs_R", pool, matrix(as.double(pairs), nrow = nrow(pairs))))
}


#' @name catboost.pool.set_pairs_weight
#' @title Set pair weights on a Pool
#' @description Set the per-pair weight of a Pool's existing pairs, keeping
#' the (winner_id, loser_id) ids unchanged.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @param pairs_weight A numeric vector of per-pair weights, length equal to
#' \code{\link{catboost.pool.num_pairs}(pool)}.
#'
#' Default value: Required argument
#' @return Nothing. Mutates \code{pool} in place.
#' @export
catboost.pool.set_pairs_weight <- function(pool, pairs_weight) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    invisible(.Call("CatBoostPoolSetPairsWeight_R", pool, as.double(pairs_weight)))
}


#' @name catboost.pool.num_pairs
#' @title Number of pairs in a Pool
#' @description Get the number of pairs in a Pool.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @return An integer, the number of pairs.
#' @export
catboost.pool.num_pairs <- function(pool) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    return(.Call("CatBoostPoolNumPairs_R", pool))
}


#' @name catboost.pool.set_timestamp
#' @title Set timestamps on a Pool
#' @description Set the per-object timestamp of a Pool.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @param timestamp A numeric vector of per-object timestamps, length equal
#' to \code{nrow(pool)}.
#'
#' Default value: Required argument
#' @return Nothing. Mutates \code{pool} in place.
#' @export
catboost.pool.set_timestamp <- function(pool, timestamp) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    invisible(.Call("CatBoostPoolSetTimestamp_R", pool, as.double(timestamp)))
}


#' @name catboost.pool.num_row
#' @title Number of rows in a Pool
#' @description Get the number of objects (rows) in a Pool.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @return The number of rows.
#' @export
catboost.pool.num_row <- function(pool) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    return(.Call("CatBoostPoolNumRow_R", pool))
}


#' @name catboost.pool.num_col
#' @title Number of columns in a Pool
#' @description Get the number of features (columns) in a Pool.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @return The number of feature columns.
#' @export
catboost.pool.num_col <- function(pool) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    return(.Call("CatBoostPoolNumCol_R", pool))
}


#' @name catboost.pool.shape
#' @title Shape of a Pool
#' @description Get the (rows, columns) shape of a Pool.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @return An integer vector of length 2: \code{c(num_row, num_col)}.
#' @export
catboost.pool.shape <- function(pool) {
    return(c(catboost.pool.num_row(pool), catboost.pool.num_col(pool)))
}


#' @name catboost.pool.is_empty
#' @title Is the Pool empty
#' @description Check whether the Pool has no objects.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @return \code{TRUE} if the Pool has zero rows, \code{FALSE} otherwise.
#' @export
catboost.pool.is_empty <- function(pool) {
    return(catboost.pool.num_row(pool) == 0)
}


#' @name catboost.pool.get_feature_names
#' @title Get feature names from a Pool
#' @description Get the names of the features (columns) of a Pool.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @return A character vector of feature names, length equal to
#' \code{catboost.pool.num_col(pool)}. Unnamed features are empty strings.
#' @export
catboost.pool.get_feature_names <- function(pool) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    return(.Call("CatBoostPoolGetFeatureNames_R", pool))
}


#' @name catboost.pool.set_feature_names
#' @title Set feature names on a Pool
#' @description Set the names of the features (columns) of a Pool.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @param feature_names A character vector of feature names, length equal to
#' \code{catboost.pool.num_col(pool)}.
#'
#' Default value: Required argument
#' @return Nothing. Mutates \code{pool} in place.
#' @export
catboost.pool.set_feature_names <- function(pool, feature_names) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    invisible(.Call("CatBoostPoolSetFeatureNames_R", pool, as.character(feature_names)))
}


#' @name catboost.pool.get_cat_feature_indices
#' @title Get categorical feature indices from a Pool
#' @description Get the (0-based) column indices of the categorical
#' features of a Pool.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @return An integer vector of 0-based categorical feature indices.
#' @export
catboost.pool.get_cat_feature_indices <- function(pool) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    return(.Call("CatBoostPoolGetCatFeatureIndices_R", pool))
}


#' @name catboost.pool.get_text_feature_indices
#' @title Get text feature indices from a Pool
#' @description Get the (0-based) column indices of the text features of a
#' Pool.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @return An integer vector of 0-based text feature indices.
#' @export
catboost.pool.get_text_feature_indices <- function(pool) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    return(.Call("CatBoostPoolGetTextFeatureIndices_R", pool))
}


#' @name catboost.pool.get_embedding_feature_indices
#' @title Get embedding feature indices from a Pool
#' @description Get the (0-based) column indices of the embedding features
#' of a Pool. Always \code{integer(0)}: this package does not support
#' building Pools with embedding features.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @return An integer vector of 0-based embedding feature indices.
#' @export
catboost.pool.get_embedding_feature_indices <- function(pool) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    return(.Call("CatBoostPoolGetEmbeddingFeatureIndices_R", pool))
}


#' @name catboost.pool.get_features
#' @title Get the feature matrix from a Pool
#' @description Get the raw numeric feature matrix of a Pool. Only
#' supported for Pools whose features are all numeric (no categorical or
#' text features).
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @return A (rows x columns) numeric matrix of feature values.
#' @export
catboost.pool.get_features <- function(pool) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    return(.Call("CatBoostPoolGetFeatures_R", pool))
}


#' @name catboost.pool.quantize
#' @title Quantize a Pool
#' @description Quantize this Pool in place: build the binarized (quantized)
#' feature data used by training, same as Python's \code{Pool.quantize()}
#' (\code{catboost/libs/data/quantization.h}'s
#' \code{ConstructQuantizedPoolFromRawPool}, the same core entry point
#' \code{catboost.train} itself uses).
#' @param pool A catboost.Pool object. Must not already be quantized.
#'
#' Default value: Required argument
#' @param params A named list of quantization parameters (e.g.
#' \code{border_count}, \code{feature_border_type}, \code{nan_mode},
#' \code{per_float_feature_quantization}, \code{ignored_features}). Same
#' names/semantics as \code{catboost.train}'s \code{params}.
#'
#' Default value: \code{list()}
#' @return Nothing. The Pool is quantized in place.
#' @export
catboost.pool.quantize <- function(pool, params = list()) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    if (catboost.pool.is_quantized(pool))
        stop("Pool is already quantized")
    params <- process_synonyms(params)
    json_params <- prepare_train_export_parameters(params)
    invisible(.Call("CatBoostPoolQuantize_R", pool, json_params))
}


#' @name catboost.pool.is_quantized
#' @title Is the Pool quantized
#' @description Check whether the Pool's feature data has already been
#' quantized (either by \code{catboost.pool.quantize} or as a side effect of
#' \code{catboost.train}).
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @return \code{TRUE} if the Pool is quantized, \code{FALSE} otherwise.
#' @export
catboost.pool.is_quantized <- function(pool) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    return(.Call("CatBoostPoolIsQuantized_R", pool))
}


#' @name catboost.pool.save_quantization_borders
#' @title Save a Pool's quantization borders to a file
#' @description Save the borders used in numeric feature quantization to a
#' file, so they can be reused to quantize another Pool identically (via
#' upstream's \code{input_borders} mechanism). File format is described at
#' \url{https://catboost.ai/docs/concepts/input-data_custom-borders.html}.
#' @param pool A catboost.Pool object. Must already be quantized.
#'
#' Default value: Required argument
#' @param output_file Output file path.
#'
#' Default value: Required argument
#' @return Nothing. Writes \code{output_file} as a side effect.
#' @export
catboost.pool.save_quantization_borders <- function(pool, output_file) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    if (!is.character(output_file))
        stop("output_file must be a string.")
    output_file <- path.expand(output_file)
    invisible(.Call("CatBoostPoolSaveQuantizationBorders_R", pool, output_file))
}


#' @name catboost.dataset_statistics
#' @title Calculate dataset statistics
#' @description R equivalent of CatBoost CLI's \code{dataset-statistics}
#' mode: computes per-feature statistics (and, unless
#' \code{only_light_statistics} is set, per-float-feature histograms) for a
#' dataset read from disk. Calls the same core library entry point the CLI
#' mode itself calls
#' (\code{catboost/private/libs/app_helpers/mode_dataset_statistics_helpers.h}'s
#' \code{NCB::CalculateDatasetStatisticsSingleHost}) directly, in-process --
#' no CLI binary shell-out.
#' @param pool_path Path to the dataset file (same format \code{catboost.from_file} reads).
#'
#' Default value: Required argument
#' @param cd_path Path to the column description file.
#'
#' Default value: \code{""} (no column description)
#' @param pairs_path Path to the pairs file.
#'
#' Default value: \code{""} (no pairs)
#' @param delimiter Column delimiter in the dataset file.
#'
#' Default value: \code{"\\t"}
#' @param has_header Whether the dataset file has a header row.
#'
#' Default value: \code{FALSE}
#' @param thread_count Number of threads to use. \code{-1} means use all cores.
#'
#' Default value: \code{-1}
#' @param border_count Number of histogram bins per float feature.
#'
#' Default value: \code{254}
#' @param only_group_statistics Only compute group-related statistics.
#'
#' Default value: \code{FALSE}
#' @param only_light_statistics Skip the second-pass histogram computation.
#'
#' Default value: \code{FALSE}
#' @return A named list with elements \code{statistics} and \code{histograms}
#' (each the parsed contents of the corresponding CLI JSON output file;
#' \code{histograms} is \code{NULL} if \code{only_light_statistics} was set).
#' @export
catboost.dataset_statistics <- function(pool_path, cd_path = "", pairs_path = "", delimiter = "\t",
                                         has_header = FALSE, thread_count = -1, border_count = 254,
                                         only_group_statistics = FALSE, only_light_statistics = FALSE) {
    if (missing(pool_path))
        stop("Need to specify pool path.")
    if (!is.character(pool_path) || !is.character(cd_path) || !is.character(pairs_path))
        stop("Path must be a string.")

    pool_path <- path.expand(pool_path)
    cd_path <- path.expand(cd_path)
    output_path <- tempfile(fileext = ".json")
    histogram_path <- tempfile(fileext = ".json")
    on.exit(unlink(c(output_path, histogram_path)))

    .Call("CatBoostDatasetStatistics_R", pool_path, cd_path, pairs_path, delimiter, has_header,
          thread_count, border_count, only_group_statistics, only_light_statistics,
          output_path, histogram_path)

    statistics <- jsonlite::fromJSON(output_path, simplifyVector = TRUE)
    histograms <- if (only_light_statistics) NULL else jsonlite::fromJSON(histogram_path, simplifyVector = TRUE)
    return(list(statistics = statistics, histograms = histograms))
}


#' @name catboost.pool.slice
#' @title Slice a Pool
#' @description Return a new Pool containing a contiguous range of rows from
#' \code{pool}: R equivalent of Python's \code{Pool.slice()}. Like Python's
#' \code{Pool.slice()} (which calls \code{_take_slice()} in
#' \code{catboost/python-package/catboost/_catboost.pyx}), this builds the new
#' Pool with the core \code{TDataProvider::GetSubset} machinery via
#' \code{CatBoostPoolSliceSubset_R} (\code{src/catboostr.cpp}), so all column
#' kinds -- numeric, categorical, text and embedding features, every target
#' column, weights, group ids, subgroup ids, baseline, pairs and feature names
#' -- are preserved in the sliced Pool.
#'
#' The only difference from Python is the row selector: Python accepts an
#' arbitrary row-index array (\code{rindex}), whereas this exposes a contiguous
#' \code{[offset, offset + size)} range.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @param offset Zero-based index of the first row to include.
#'
#' Default value: Required argument
#' @param size Number of rows to include.
#'
#' Default value: Required argument
#' @return A new catboost.Pool object containing the sliced rows.
#' @export
catboost.pool.slice <- function(pool, offset, size) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    if (!is.numeric(offset) || length(offset) != 1 || offset < 0)
        stop("offset must be a single non-negative number.")
    if (!is.numeric(size) || length(size) != 1 || size < 0)
        stop("size must be a single non-negative number.")

    sliced <- .Call("CatBoostPoolSliceSubset_R", pool, as.integer(size), as.integer(offset))
    attributes(sliced) <- attributes(pool)
    return(sliced)
}


#' @name catboost.pool.train_eval_split
#' @title Split a Pool into train/eval subsets
#' @description R equivalent of Python's \code{Pool.train_eval_split()}:
#' splits \code{pool} into a train and (optionally) an eval Pool, with the
#' same shuffle-then-split(-then-stratify) semantics as the Python method.
#' Python's own implementation (\code{TrainEvalSplit()} in
#' \code{catboost/python-package/catboost/helpers.cpp}) lives in the
#' python-package tree (not a core lib) and includes \code{Python.h}, so it
#' is not directly callable from R; \code{CatBoostPoolTrainEvalSplit_R}
#' (\code{src/catboostr.cpp}) reimplements its body against the same core
#' entry points it itself calls.
#' @param pool A catboost.Pool object.
#'
#' Default value: Required argument
#' @param has_time If \code{TRUE}, disables shuffling before the split
#' (preserves row order, e.g. for time-ordered data).
#'
#' Default value: \code{FALSE}
#' @param is_classification If \code{TRUE}, performs a stratified split using
#' the Pool's label as the class column.
#'
#' Default value: \code{FALSE}
#' @param eval_fraction Fraction of rows (in \code{(0, 1)}) to hold out for
#' eval.
#'
#' Default value: \code{0.2}
#' @param save_eval_pool If \code{FALSE}, the eval Pool is not built and
#' \code{eval} is \code{NULL} in the returned list.
#'
#' Default value: \code{TRUE}
#' @return A named list with elements \code{train} and \code{eval} (each a
#' catboost.Pool, or \code{NULL} for \code{eval} if \code{save_eval_pool} is
#' \code{FALSE}).
#' @export
catboost.pool.train_eval_split <- function(pool, has_time = FALSE, is_classification = FALSE,
                                            eval_fraction = 0.2, save_eval_pool = TRUE) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    if (!is.numeric(eval_fraction) || length(eval_fraction) != 1 || eval_fraction <= 0 || eval_fraction >= 1)
        stop("eval_fraction must be a single number in (0, 1).")

    result <- .Call("CatBoostPoolTrainEvalSplit_R", pool, has_time, is_classification, eval_fraction, save_eval_pool)

    train_pool <- result[[1]]
    attributes(train_pool) <- attributes(pool)

    eval_pool <- NULL
    if (save_eval_pool) {
        eval_pool <- result[[2]]
        attributes(eval_pool) <- attributes(pool)
    }

    return(list(train = train_pool, eval = eval_pool))
}


#' @name catboost.pool.save
#' @title Save a quantized Pool to CatBoost's binary quantized-pool format
#' @description R equivalent of Python's \code{Pool.save()}: saves an
#' already-quantized Pool to CatBoost's own binary quantized-pool format
#' (\code{catboost/private/libs/quantized_pool/serialization.h}'s
#' \code{SaveQuantizedPool}, the same core entry point Python's
#' \code{Pool.save()}/\code{_save()} calls). This is a different, binary
#' format from \code{catboost.save_pool} (R/catboost.R), which writes CD/TSV
#' files read back by \code{catboost.load_pool}'s \code{column_description}
#' path -- \code{catboost.save_pool} does NOT already satisfy Python's
#' \code{Pool.save()}.
#' @param pool A catboost.Pool object. Must already be quantized (see
#' \code{catboost.pool.quantize}).
#'
#' Default value: Required argument
#' @param fname Output file path.
#'
#' Default value: Required argument
#' @return Nothing. Writes \code{fname} as a side effect.
#' @export
catboost.pool.save <- function(pool, fname) {
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    if (!is.character(fname))
        stop("fname must be a string.")
    fname <- path.expand(fname)
    invisible(.Call("CatBoostPoolSave_R", pool, fname))
}


#' @title Print basic information about model
#' @description Displays the most general characteristics of a CatBoost model.
#' @param x The model obtained as the result of training.
#' @param ... Not used
#' @return The same model that was passed as input.
#' @export
print.catboost.Model <- function(x, ...) {
    cat(sprintf("CatBoost model (%d trees)\n", x$tree_count))
    cat(sprintf("Loss function: %s\n", catboost.get_plain_params(x)$loss_function))
    cat(sprintf("Fit to %d feature(s)\n", x$feature_count))
    if (is.null.handle(x$cpp_obj$handle))
        cat("(Handle is incomplete)\n")
    return(invisible(x))
}


#' @title Print basic information about model
#' @description Displays the most general characteristics of a CatBoost model
#' (same as 'print').
#' @param object The model obtained as the result of training.
#' @param ... Not used
#' @return The same model that was passed as input.
#' @export
summary.catboost.Model <- function(object, ...) {
    print.catboost.Model(object)
}

# P6.5 (catboost-8z4.94): custom_objective's params$loss_function
# validation/defaulting, factored out of catboost.train (catboost-8z4.89)
# now that catboost.cv/grid_search/randomized_search/eval_feature need the
# identical check before serializing their own params JSON (5 more call
# sites -- past the "3+ occurrences" duplication threshold). Returns params
# unchanged when custom_objective is NULL. Not roxygen-documented (internal
# helper, not exported); placed above catboost.train's own docblock (rather
# than directly above catboost.train's definition) so roxygen2 attaches that
# docblock to catboost.train and not to this function.
apply_custom_objective_params <- function(params, custom_objective) {
    if (is.null(custom_objective)) {
        return(params)
    }
    if (!is.list(custom_objective))
        stop("'custom_objective' must be a list with 'calc_ders_range' and/or 'calc_ders_multi' function elements, got: ", class(custom_objective))
    has_calc_ders_range <- is.function(custom_objective$calc_ders_range)
    has_calc_ders_multi <- is.function(custom_objective$calc_ders_multi)
    if (!has_calc_ders_range && !has_calc_ders_multi)
        stop("'custom_objective' must define at least one of 'calc_ders_range' or 'calc_ders_multi' as a function")

    # Mirrors Python's _PreprocessParams (_catboost.pyx): a custom objective
    # is dispatched to TCustomError/TMultiTargetCustomError
    # (vendor/catboost/catboost/private/libs/algo/tensor_search_helpers.cpp)
    # purely by loss_function's *value*, not by any separate flag, so this
    # value must be one of these two markers whenever custom_objective is
    # supplied. "PythonUserDefinedPerObject" covers both calc_ders_range
    # (single-dimension) and calc_ders_multi used for MultiClass-shaped
    # losses (scalar target); set loss_function to
    # "PythonUserDefinedMultiTarget" explicitly yourself if calc_ders_multi
    # implements a multi-target regression objective (vector target) instead.
    if (is.null(params$loss_function)) {
        params$loss_function <- "PythonUserDefinedPerObject"
    } else if (!(params$loss_function %in% c("PythonUserDefinedPerObject", "PythonUserDefinedMultiTarget"))) {
        stop("'loss_function' must be \"PythonUserDefinedPerObject\" or \"PythonUserDefinedMultiTarget\" ",
             "when 'custom_objective' is supplied (got: ", params$loss_function, "). ",
             "Leave 'loss_function' unset to default to \"PythonUserDefinedPerObject\".")
    }
    return(params)
}

# P6.5 (catboost-8z4.94): custom_eval_metric_object's params$eval_metric
# validation/defaulting, factored out of catboost.train (catboost-8z4.90) --
# see apply_custom_objective_params above for why (including the
# not-roxygen-documented / placement rationale).
apply_custom_eval_metric_params <- function(params, custom_eval_metric_object) {
    if (is.null(custom_eval_metric_object)) {
        return(params)
    }
    if (!is.list(custom_eval_metric_object))
        stop("'custom_eval_metric_object' must be a list with 'evaluate' and 'is_max_optimal' function elements, got: ", class(custom_eval_metric_object))
    if (!is.function(custom_eval_metric_object$evaluate))
        stop("'custom_eval_metric_object' must define an 'evaluate' function")
    if (!is.function(custom_eval_metric_object$is_max_optimal))
        stop("'custom_eval_metric_object' must define an 'is_max_optimal' function")

    # Mirrors Python's own validation (_catboost.pyx's _PreprocessParams):
    # dispatch to TCustomMetric/TMultiTargetCustomMetric
    # (vendor/catboost/catboost/libs/metrics/metric.cpp) is driven purely by
    # which of EvalFunc/EvalMultiTargetFunc this package wires (from
    # 'multi_target' below), but eval_metric's *value* must still be one of
    # these two markers for native CreateMetrics() to route to a custom
    # metric at all (IsUserDefined() check).
    is_multi_target_metric <- isTRUE(custom_eval_metric_object$multi_target)
    expected_eval_metric <- if (is_multi_target_metric) "PythonUserDefinedMultiTarget" else "PythonUserDefinedPerObject"
    if (is.null(params$eval_metric)) {
        params$eval_metric <- expected_eval_metric
    } else if (!identical(params$eval_metric, expected_eval_metric)) {
        stop("'eval_metric' must be \"", expected_eval_metric, "\" when 'custom_eval_metric_object' is supplied ",
             "with multi_target = ", is_multi_target_metric, " (got: ", params$eval_metric, "). ",
             "Leave 'eval_metric' unset to default to \"", expected_eval_metric, "\".")
    }
    return(params)
}


#' @name catboost.train
#' @title Train the model
#'
#' @description Train the model using a CatBoost dataset.
#'
#' The list of parameters
#'
#' \itemize{
#'   \item Common parameters
#'   \itemize{
#'     \item fold_permutation_block
#'
#'       Objects in the dataset are grouped in blocks before the random permutations.
#'       This parameter defines the size of the blocks.
#'       The smaller is the value, the slower is the training.
#'       Large values may result in quality degradation.
#'
#'       Default value:
#'
#'       Default value differs depending on the dataset size and ranges from 1 to 256 inclusively
#'     \item ignored_features
#'
#'       Identifiers of features to exclude from training.
#'       The non-negative indices that do not match any features are successfully ignored.
#'       For example, if five features are defined for the objects in the dataset and this parameter
#'       is set to "42", the corresponding non-existing feature is successfully ignored.
#'
#'       The identifier corresponds to the feature's index.
#'       Feature indices used in train and feature importance are numbered from 0 to featureCount-1.
#'       If a file is used as input data then any non-feature column types are ignored when calculating these
#'       indices. For example, each row in the input file contains data in the following order:
#'       "categorical feature<\verb{\t}>label<\verb{\t}>numerical feature". So for the row "rock<\verb{\t}>0<\verb{\t}>42",
#'       the identifier for the "rock" feature is 0, and for the "42" feature it is 1.
#'
#'       The identifiers of features to exclude should be enumerated at vector.
#'
#'       For example, if training should exclude features with the identifiers
#'       1, 2, 7, 42, 43, 44, 45, the value of this parameter should be set to c(1,2,7,42,43,44,45).
#'
#'       Default value:
#'
#'       None (use all features)
#'     \item use_best_model
#'
#'       If this parameter is set, the number of trees that are saved in the resulting model is defined as follows:
#'
#'       Build the number of trees defined by the training parameters.
#'       \itemize{
#'         \item Identify the iteration with the optimal loss function value.
#'         \item No trees are saved after this iteration.
#'       }
#'
#'       This option requires a test dataset to be provided.
#'
#'       Default value:
#'
#'       FALSE (not used)
#'     \item loss_function
#'
#'       The loss function (see \url{https://catboost.ai/docs/concepts/loss-functions.html#loss-functions})
#'       to use in training. The specified value also determines the machine learning problem to solve.
#'
#'       Format:
#'
#'       <Loss function 1>[:<parameter 1>=<value>:..<parameter N>=<value>:]
#'
#'       Supported loss functions:
#'       \itemize{
#'         \item 'Logloss'
#'         \item 'CrossEntropy'
#'         \item 'MultiClass'
#'         \item 'MultiClassOneVsAll'
#'         \item 'RMSE'
#'         \item 'MAE'
#'         \item 'Quantile'
#'         \item 'LogLinQuantile'
#'         \item 'MAPE'
#'         \item 'Poisson'
#'         \item 'Lq'
#'         \item 'PairLogit'
#'         \item 'PairLogitPairwise'
#'         \item 'YetiRank'
#'         \item 'YetiRankPairwise'
#'         \item 'QueryCrossEntropy'
#'         \item 'QueryRMSE'
#'         \item 'QuerySoftMax'
#'       }
#'
#'       Supported parameters:
#'       \itemize{
#'         \item alpha - The coefficient used in quantile-based losses ('Quantile' and 'LogLinQuantile'). The default value is 0.5.
#'
#'         For example, if you need to calculate the value of Quantile with the coefficient \eqn{\alpha = 0.1}, use the following construction:
#'
#'         'Quantile:alpha=0.1'
#'       }
#'
#'       Default value:
#'
#'       'RMSE'
#'     \item custom_loss
#'
#'       Loss function (see \url{https://catboost.ai/docs/concepts/loss-functions.html#loss-functions})
#'       values to output during training.
#'       These functions are not used for optimization and are displayed for informational purposes only.
#'
#'       Format:
#'
#'       c(<Loss function 1>[:<parameter>=<value>],<Loss function 2>[:<parameter>=<value>],...,<Loss function N>[:<parameter>=<value>])
#'
#'       Supported loss functions:
#'       \itemize{
#'         \item 'Logloss'
#'         \item 'CrossEntropy'
#'         \item 'Precision'
#'         \item 'Recall'
#'         \item 'F1'
#'         \item 'F'
#'         \item 'BalancedAccuracy'
#'         \item 'BalancedErrorRate'
#'         \item 'MCC'
#'         \item 'Accuracy'
#'         \item 'CtrFactor'
#'         \item 'AUC'
#'         \item 'BrierScore'
#'         \item 'HingeLoss'
#'         \item 'HammingLoss'
#'         \item 'ZeroOneLoss'
#'         \item 'Kappa'
#'         \item 'WKappa'
#'         \item 'LogLikelihoodOfPrediction'
#'         \item 'MultiClass'
#'         \item 'MultiClassOneVsAll'
#'         \item 'TotalF1'
#'         \item 'MAE'
#'         \item 'MAPE'
#'         \item 'Poisson'
#'         \item 'Quantile'
#'         \item 'RMSE'
#'         \item 'LogLinQuantile'
#'         \item 'Lq'
#'         \item 'NumErrors'
#'         \item 'SMAPE'
#'         \item 'R2'
#'         \item 'MSLE'
#'         \item 'MedianAbsoluteError'
#'         \item 'PairLogit'
#'         \item 'PairLogitPairwise'
#'         \item 'PairAccuracy'
#'         \item 'QueryCrossEntropy'
#'         \item 'QueryRMSE'
#'         \item 'QuerySoftMax'
#'         \item 'PFound'
#'         \item 'NDCG'
#'         \item 'AverageGain'
#'         \item 'PrecisionAt'
#'         \item 'RecallAt'
#'         \item 'MAP'
#'         \item 'MRR'
#'         \item 'ERR'
#'       }
#'
#'       Supported parameters:
#'       \itemize{
#'         \item alpha - The coefficient used in quantile-based losses ('Quantile' and 'LogLinQuantile'). The default value is 0.5.
#'       }
#'
#'       For example, if you need to calculate the value of CrossEntropy and Quantile with the coefficient \eqn{\alpha = 0.1}, use the following construction:
#'
#'       c('CrossEntropy') or simply 'CrossEntropy'.
#'
#'       Values of all custom loss functions for learning and test datasets are saved to the Loss function
#'       (see \url{https://catboost.ai/docs/concepts/output-data_loss-function.html#output-data_loss-function})
#'       output files (learn_error.tsv and test_error.tsv respectively). The catalog for these files is specified in the train-dir (train_dir) parameter.
#'
#'       Default value:
#'
#'       None (use one of the loss functions supported by the library)
#'     \item eval_metric
#'
#'       The loss function used for overfitting detection (if enabled) and best model selection (if enabled).
#'
#'       Supported loss functions:
#'       \itemize{
#'         \item 'Logloss'
#'         \item 'CrossEntropy'
#'         \item 'Precision'
#'         \item 'Recall'
#'         \item 'F1'
#'         \item 'F'
#'         \item 'BalancedAccuracy'
#'         \item 'BalancedErrorRate'
#'         \item 'MCC'
#'         \item 'Accuracy'
#'         \item 'CtrFactor'
#'         \item 'AUC'
#'         \item 'BrierScore'
#'         \item 'HingeLoss'
#'         \item 'HammingLoss'
#'         \item 'ZeroOneLoss'
#'         \item 'Kappa'
#'         \item 'WKappa'
#'         \item 'LogLikelihoodOfPrediction'
#'         \item 'MultiClass'
#'         \item 'MultiClassOneVsAll'
#'         \item 'TotalF1'
#'         \item 'MAE'
#'         \item 'MAPE'
#'         \item 'Poisson'
#'         \item 'Quantile'
#'         \item 'RMSE'
#'         \item 'LogLinQuantile'
#'         \item 'Lq'
#'         \item 'NumErrors'
#'         \item 'SMAPE'
#'         \item 'R2'
#'         \item 'MSLE'
#'         \item 'MedianAbsoluteError'
#'         \item 'PairLogit'
#'         \item 'PairLogitPairwise'
#'         \item 'PairAccuracy'
#'         \item 'QueryCrossEntropy'
#'         \item 'QueryRMSE'
#'         \item 'QuerySoftMax'
#'         \item 'PFound'
#'         \item 'NDCG'
#'         \item 'AverageGain'
#'         \item 'PrecisionAt'
#'         \item 'RecallAt'
#'         \item 'MAP'
#'         \item 'MRR'
#'         \item 'ERR'
#'       }
#'
#'       Format:
#'
#'       metric_name:param=Value
#'
#'       Examples:
#'
#'       \code{'R2'}
#'
#'       \code{'Quantile:alpha=0.3'}
#'
#'       Default value:
#'
#'       Optimized objective is used
#'
#'     \item iterations
#'
#'       The maximum number of trees that can be built when solving machine learning problems.
#'
#'       When using other parameters that limit the number of iterations, the final number of trees may be less
#'       than the number specified in this parameter.
#'
#'       Default value:
#'
#'       1000
#'
#'     \item border
#'
#'       The target border. If the value is strictly greater than this threshold,
#'       it is considered a positive class. Otherwise it is considered a negative class.
#'
#'       The parameter is obligatory if the Logloss function is used, since it uses borders to transform
#'       any given target to a binary target.
#'
#'       Used in binary classification.
#'
#'       Default value:
#'
#'       0.5
#'
#'     \item leaf_estimation_iterations
#'
#'       The number of gradient steps when calculating the values in leaves.
#'
#'       Default value:
#'
#'       1
#'
#'     \item depth
#'
#'       Depth of the trees.
#'
#'       The value can be any integer up to 16. It is recommended to use values in the range [1; 10].
#'
#'       Default value:
#'
#'       6
#'     \item learning_rate
#'
#'       The learning rate.
#'
#'       Used for reducing the gradient step.
#'
#'       Default value:
#'
#'       0.03
#'
#'     \item rsm
#'
#'       Random subspace method. The percentage of features to use at each iteration of building trees. At each iteration, features are selected over again at random.
#'
#'       The value must be in the range [0;1].
#'
#'       Default value:
#'
#'       1
#'
#'     \item random_seed
#'
#'       The random seed used for training.
#'
#'       Default value:
#'
#'       0
#'
#'    \item nan_mode
#'
#'       Way to process missing values.
#'
#'       Possible values:
#'       \itemize{
#'         \item \code{'Min'}
#'         \item \code{'Max'}
#'         \item \code{'Forbidden'}
#'       }
#'
#'       Default value:
#'
#'       \code{'Min'}
#'
#'     \item od_pval
#'
#'       Use the Overfitting detector (see \url{https://catboost.ai/docs/concepts/overfitting-detector.html#overfitting-detector})
#'       to stop training when the threshold is reached.
#'       Requires that a test dataset was input.
#'
#'       For best results, it is recommended to set a value in the range [10^-10; 10^-2].
#'
#'       The larger the value, the earlier overfitting is detected.
#'
#'       Default value:
#'
#'       The overfitting detection is turned off
#'
#'     \item od_type
#'
#'       The method used to calculate the values in leaves.
#'
#'       Possible values:
#'       \itemize{
#'         \item IncToDec
#'         \item Iter
#'       }
#'
#'       Restriction.
#'       Do not specify the overfitting detector threshold when using the Iter type.
#'
#'       Default value:
#'
#'       'IncToDec'
#'
#'     \item od_wait
#'
#'       The number of iterations to continue the training after the iteration with the optimal loss function value.
#'       The purpose of this parameter differs depending on the selected overfitting detector type:
#'       \itemize{
#'         \item IncToDec - Ignore the overfitting detector when the threshold is reached and continue learning for the specified number of iterations after the iteration with the optimal loss function value.
#'         \item Iter - Consider the model overfitted and stop training after the specified number of iterations since the iteration with the optimal loss function value.
#'       }
#'
#'       Default value:
#'
#'       20
#'
#'     \item leaf_estimation_method
#'
#'       The method used to calculate the values in leaves.
#'
#'       Possible values:
#'       \itemize{
#'         \item Newton
#'         \item Gradient
#'       }
#'
#'       Default value:
#'
#'       Default value depends on the selected loss function
#'
#'     \item grow_policy
#'
#'       GPU only. The tree growing policy. It describes how to perform greedy tree construction.
#'
#'       Possible values:
#'       \itemize{
#'         \item SymmetricTree
#'         \item Lossguide
#'         \item Depthwise
#'       }
#'
#'       Default value:
#'
#'       SymmetricTree
#'
#'     \item min_data_in_leaf
#'
#'       GPU only.
#'       The minimum training samples count in leaf.
#'       CatBoost will not search for new splits in leaves with samples count less than min_data_in_leaf.
#'       This parameter is used only for Depthwise and Lossguide growing policies.
#'
#'       Default value:
#'
#'       1
#'
#'     \item max_leaves
#'
#'       GPU only. The maximum leaf count in resulting tree. Used only for Lossguide growing policy.
#'       This parameter is used only for Lossguide growing policy.
#'
#'       Default value:
#'
#'       31
#'
#'     \item score_function
#'       GPU only. Score that is used during tree construction to select the next tree split.
#'
#'       Possible values:
#'       \itemize{
#'         \item L2
#'         \item Cosine
#'         \item NewtonL2
#'         \item NewtonCosine
#'       }
#'
#'       Default value:
#'
#'       Cosine
#'
#'       For growing policy Lossguide default is NewtonL2.
#'
#'     \item l2_leaf_reg
#'
#'       L2 regularization coefficient. Used for leaf value calculation.
#'
#'       Any positive values are allowed.
#'
#'       Default value:
#'
#'       3
#'
#'     \item model_size_reg
#'
#'       Model size regularization coefficient. The influence coefficient of the model size for choosing tree structure.
#'       To get a smaller model size - increase this coefficient.
#'
#'       Any positive values are allowed.
#'
#'       Default value:
#'
#'       0.5
#'
#'     \item has_time
#'
#'       Use the order of objects in the input data
#'       (do not perform a random permutation of the dataset at the preprocessing stage)
#'
#'       Default value:
#'
#'       FALSE (not used; permute input dataset)
#'
#'     \item allow_const_label
#'
#'       To allow the constant label value in the dataset.
#'
#'       Default value:
#'
#'       FALSE
#'
#'     \item name
#'
#'       The experiment name to display in visualization tools (see \url{https://catboost.ai/docs/features/visualization.html#visualization}).
#'
#'       Default value:
#'
#'       experiment
#'
#'     \item prediction_type
#'
#'       The format for displaying approximated values in output data.
#'
#'       Possible values:
#'       \itemize{
#'         \item 'Probability'
#'         \item 'Class'
#'         \item 'RawFormulaVal'
#'       }
#'
#'       Default value:
#'
#'       \code{'RawFormulaVal'}
#'
#'     \item fold_len_multiplier
#'
#'       Coefficient for changing the length of folds.
#'
#'       The value must be greater than 1. The best validation result is achieved with minimum values.
#'
#'       With values close to 1 (for example, \eqn{1 + \epsilon}), each iteration takes a quadratic amount of memory and time
#'       for the number of objects in the iteration. Thus, low values are possible only when there is a small number of objects.
#'
#'       Default value:
#'
#'       2
#'
#'     \item class_weights
#'
#'       Classes weights. The values are used as multipliers for the object weights.
#'
#'       For example, for 3 class classification you could use:
#'
#'       \code{c(0.85, 1.2, 1)}
#'
#'       Default value:
#'
#'       None (the weight for all classes is set to 1)
#'
#'     \item classes_count
#'
#'       The upper limit for the numeric class label. Defines the number of classes for multiclassification.
#'
#'       Only non-negative integers can be specified. The given integer should be greater than any of the target
#'       values.
#'
#'       If this parameter is specified the labels for all classes in the input dataset should be smaller
#'       than the given value.
#'
#'       Default value:
#'
#'       maximum class label + 1
#'
#'     \item one_hot_max_size
#'
#'       Convert the feature to float if the number of different values that it takes exceeds the specified value. Ctrs are not calculated for such features.
#'
#'       The one-vs.-all delimiter is used for the resulting float features.
#'
#'       Default value:
#'
#'       FALSE
#'
#'       Do not convert features to float based on the number of different values
#'
#'     \item random_strength
#'
#'       Score standard deviation multiplier.
#'
#'        Default value:
#'
#'        1
#'
#'     \item bootstrap_type
#'
#'       Bootstrap type. Defines the method for sampling the weights of documents.
#'
#'       Possible values:
#'       \itemize{
#'         \item 'Bayesian'
#'         \item 'Bernoulli'
#'         \item 'Poisson'
#'         \item 'MVS'
#'         \item 'No'
#'       }
#'
#'       Poisson bootstrap is supported only on GPU.
#'
#'       Default value:
#'
#'       \code{'Bayesian'}
#'
#'     \item bagging_temperature
#'
#'        Controls intensity of Bayesian bagging. The higher the temperature the more aggressive bagging is.
#'
#'        Typical values are in the range \eqn{[0, 1]} (0 is for no bagging).
#'
#'        Possible values are in the range \eqn{[0, +\infty)}.
#'
#'        Default value:
#'
#'        1
#'
#'     \item subsample
#'
#'       Sample rate for bagging. This parameter can be used if one of the following bootstrap types is defined:
#'       \itemize{
#'         \item 'Bernoulli'
#'       }
#'
#'       Default value:
#'
#'       0.66
#'
#'     \item sampling_unit
#'
#'       The parameter allows to specify the sampling scheme: sample weights for each object individually or for an entire group of objects together.
#'
#'       Possible values:
#'       \itemize{
#'         \item 'Object'
#'         \item 'Group'
#'       }
#'
#'       Default value:
#'
#'       \code{'Object'}
#'
#'     \item sampling_frequency
#'
#'       Frequency to sample weights and objects when building trees.
#'
#'       Possible values:
#'       \itemize{
#'         \item 'PerTree'
#'         \item 'PerTreeLevel'
#'       }
#'
#'       Default value:
#'
#'       \code{'PerTreeLevel'}
#'
#'     \item model_shrink_rate
#'
#'       For i > 0 at the start of i-th iteration multiplies model by (1 - model_shrink_rate / i).
#'
#'       Possible values: [0, 1).
#'
#'       Default value: 0
#'   }
#'   \item CTR settings
#'   \itemize{
#'     \item simple_ctr
#'
#'       Binarization settings for categorical features (see \url{https://catboost.ai/docs/concepts/algorithm-main-stages_cat-to-numberic.html}).
#'
#'       Format:
#'
#'       \code{c(CtrType[:TargetBorderCount=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N])}
#'
#'       Components:
#'       \itemize{
#'         \item CTR types for training on CPU:
#'         \itemize{
#'           \item \code{'Borders'}
#'           \item \code{'Buckets'}
#'           \item \code{'BinarizedTargetMeanValue'}
#'           \item \code{'Counter'}
#'         }
#'         \item CTR types for training on GPU:
#'         \itemize{
#'           \item \code{'Borders'}
#'           \item \code{'Buckets'}
#'           \item \code{'FeatureFreq'}
#'           \item \code{'FloatTargetMeanValue'}
#'         }
#'         \item The number of borders for label value binarization. (see \url{https://catboost.ai/docs/concepts/quantization.html})
#'         Only used for regression problems. Allowed values are integers from 1 to 255 inclusively. The default value is 1.
#'         This option is available for training on CPU only.
#'         \item The binarization (see \url{https://catboost.ai/docs/concepts/quantization.html})
#'         type for the label value. Only used for regression problems.
#'
#'         Possible values:
#'         \itemize{
#'           \item \code{'Median'}
#'           \item \code{'Uniform'}
#'           \item \code{'UniformAndQuantiles'}
#'           \item \code{'MaxLogSum'}
#'           \item \code{'MinEntropy'}
#'           \item \code{'GreedyLogSum'}
#'         }
#'         By default, \code{'MinEntropy'}
#'         This option is available for training on CPU only.
#'         \item The number of splits for categorical features. Allowed values are integers from 1 to 255 inclusively.
#'         \item The binarization type for categorical features.
#'         Supported values for training on CPU:
#'         \itemize{
#'           \item \code{'Uniform'}
#'         }
#
#'         Supported values for training on GPU:
#'         \itemize{
#'           \item \code{'Median'}
#'           \item \code{'Uniform'}
#'           \item \code{'UniformAndQuantiles'}
#'           \item \code{'MaxLogSum'}
#'           \item \code{'MinEntropy'}
#'           \item \code{'GreedyLogSum'}
#'         }
#'         \item Priors to use during training (several values can be specified)
#
#'         Possible formats:
#'         \itemize{
#'           \item \code{'One number - Adds the value to the numerator.'}
#'           \item \code{'Two slash-delimited numbers (for GPU only) - Use this format to set a fraction. The number is added to the numerator and the second is added to the denominator.'}
#'         }
#'       }
#'
#'     \item combinations_ctr
#'
#'       Binarization settings for combinations of categorical features (see \url{https://catboost.ai/docs/concepts/algorithm-main-stages_cat-to-numberic.html}).
#'
#'       Format:
#'
#'       \code{c(CtrType[:TargetBorderCount=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N])}
#'
#'       Components:
#'       \itemize{
#'         \item CTR types for training on CPU:
#'         \itemize{
#'           \item \code{'Borders'}
#'           \item \code{'Buckets'}
#'           \item \code{'BinarizedTargetMeanValue'}
#'           \item \code{'Counter'}
#'         }
#'         \item CTR types for training on GPU:
#'         \itemize{
#'           \item \code{'Borders'}
#'           \item \code{'Buckets'}
#'           \item \code{'FeatureFreq'}
#'           \item \code{'FloatTargetMeanValue'}
#'         }
#'         \item The number of borders for target binarization. (see \url{https://catboost.ai/docs/concepts/quantization.html})
#'         Only used for regression problems. Allowed values are integers from 1 to 255 inclusively. The default value is 1.
#'         This option is available for training on CPU only.
#'         \item The binarization (see \url{https://catboost.ai/docs/concepts/quantization.html})
#'         type for the target. Only used for regression problems.
#'
#'         Possible values:
#'         \itemize{
#'           \item \code{'Median'}
#'           \item \code{'Uniform'}
#'           \item \code{'UniformAndQuantiles'}
#'           \item \code{'MaxLogSum'}
#'           \item \code{'MinEntropy'}
#'           \item \code{'GreedyLogSum'}
#'         }
#'         By default, \code{'MinEntropy'}
#'         This option is available for training on CPU only.
#'         \item The number of splits for categorical features. Allowed values are integers from 1 to 255 inclusively.
#'         \item The binarization type for categorical features.
#'         Supported values for training on CPU:
#'         \itemize{
#'           \item \code{'Uniform'}
#'         }
#
#'         Supported values for training on GPU:
#'         \itemize{
#'           \item \code{'Median'}
#'           \item \code{'Uniform'}
#'           \item \code{'UniformAndQuantiles'}
#'           \item \code{'MaxLogSum'}
#'           \item \code{'MinEntropy'}
#'           \item \code{'GreedyLogSum'}
#'         }
#'         \item Priors to use during training (several values can be specified)
#
#'         Possible formats:
#'         \itemize{
#'           \item \code{'One number - Adds the value to the numerator.'}
#'           \item \code{'Two slash-delimited numbers (for GPU only) - Use this format to set a fraction. The number is added to the numerator and the second is added to the denominator.'}
#'         }
#'       }
#'
#'     \item ctr_target_border_count
#'
#'       Maximum number of borders used in target binarization for categorical features that need it.
#'       If TargetBorderCount is specified in 'simple_ctr', 'combinations_ctr' or 'per_feature_ctr' option it overrides this value.
#'
#'       Default value:
#'
#'       1
#'
#'     \item counter_calc_method
#'
#'       The method for calculating the Counter CTR type for the test dataset.
#'
#'       Possible values:
#'         \itemize{
#'           \item \code{'Full'}
#'           \item \code{'FullTest'}
#'           \item \code{'PrefixTest'}
#'           \item \code{'SkipTest'}
#'         }
#'
#'         Default value: \code{'PrefixTest'}
#'
#'     \item max_ctr_complexity
#'
#'       The maximum number of categorical features that can be combined.
#'
#'       Default value:
#'
#'       4
#'
#'     \item ctr_leaf_count_limit
#'
#'       The maximum number of leaves with categorical features.
#'       If the number of leaves exceeds the specified limit, some leaves are discarded.
#'       The value must be positive (for zero limit use \code{ignored_features} parameter).
#'
#'       The leaves to be discarded are selected as follows:
#'       \enumerate{
#'         \item The leaves are sorted by the frequency of the values.
#'         \item The top N leaves are selected, where N is the value specified in the parameter.
#'         \item All leaves starting from N+1 are discarded.
#'       }
#'
#'       This option reduces the resulting model size and the amount of memory required for training.
#'       Note that the resulting quality of the model can be affected.
#'
#'       Default value:
#'
#'       None (The number of leaves with categorical features is not limited)
#'
#'     \item store_all_simple_ctr
#'
#'       Ignore categorical features, which are not used in feature combinations,
#'       when choosing candidates for exclusion.
#'
#'       Use this parameter with ctr-leaf-count-limit only.
#'
#'       Default value:
#'
#'       FALSE (Both simple features and feature combinations are taken in account when limiting the number of leaves with categorical features)
#'
#'
#'   }
#'   \item Binarization settings
#'   \itemize{
#'     \item  border_count
#'
#'       The number of splits for numerical features. Allowed values are integers from 1 to 255 inclusively.
#'
#'       Default value:
#'
#'       254 for training on CPU or 128 for training on GPU
#'
#'     \item feature_border_type
#'
#'       The binarization mode (see \url{https://catboost.ai/docs/concepts/quantization.html})
#'       for numerical features.
#'
#'       Possible values:
#'       \itemize{
#'         \item \code{'Median'}
#'         \item \code{'Uniform'}
#'         \item \code{'UniformAndQuantiles'}
#'         \item \code{'MaxLogSum'}
#'         \item \code{'MinEntropy'}
#'         \item \code{'GreedyLogSum'}
#'       }
#'
#'       Default value:
#'
#'       \code{'MinEntropy'}
#'   }
#'   \item Performance settings
#'   \itemize{
#'     \item thread_count
#'
#'       The number of threads to use when applying the model.
#'
#'       Allows you to optimize the speed of execution. This parameter doesn't affect results.
#'
#'       Default value:
#'
#'       The number of CPU cores.
#'   }
#'   \item Output settings
#'   \itemize{
#'     \item logging_level
#'
#'       Possible values:
#'       \itemize{
#'         \item \code{'Silent'}
#'         \item \code{'Verbose'}
#'         \item \code{'Info'}
#'         \item \code{'Debug'}
#'       }
#'
#'       Default value:
#'
#'       'Silent'
#'
#'     \item metric_period
#'
#'       The frequency of iterations to print the information to stdout. The value should be a positive integer.
#'
#'       Default value:
#'
#'       1
#'
#'     \item train_dir
#'
#'       The directory for storing the files generated during training.
#'
#'       Default value:
#'
#'       None (current catalog)
#'
#'     \item save_snapshot
#'
#'       Enable snapshotting for restoring the training progress after an interruption.
#'
#'       Default value:
#'
#'       None
#'
#'     \item snapshot_file
#'
#'       Settings for recovering training after an interruption (see
#'       \url{https://catboost.ai/docs/features/snapshots.html}).
#'
#'       Depending on whether the file specified exists in the file system:
#'       \itemize{
#'         \item Missing - write information about training progress to the specified file.
#'         \item Exists - load data from the specified file and continue training from where it left off.
#'       }
#'
#'       Default value:
#'
#'       File can't be generated or read. If the value is omitted, the file name is experiment.cbsnapshot.
#'
#'   \item snapshot_interval
#'
#'       Interval between saving snapshots (seconds)
#'
#'       Default value:
#'
#'       600
#'
#'   \item allow_writing_files
#'
#'       If this flag is set to FALSE, no files with different diagnostic info will be created during training.
#'       With this flag set to FALSE no snapshotting can be done. Plus visualisation will not
#'       work, because visualisation uses files that are created and updated during training.
#'
#'       Default value:
#'
#'       TRUE
#'
#'   \item approx_on_full_history
#'
#'       If this flag is set to TRUE, each approximated value is calculated using all the preceding rows in the fold (slower, more accurate).
#'       If this flag is set to FALSE, each approximated value is calculated using only the beginning 1/fold_len_multiplier fraction of the fold (faster, slightly less accurate).
#'
#'       Default value:
#'
#'       FALSE
#'
#'   \item boosting_type
#'
#'       Boosting scheme.
#'      Possible values:
#'          - 'Ordered' - Gives better quality, but may slow down the training.
#'          - 'Plain' - The classic gradient boosting scheme. May result in quality degradation, but does not slow down the training.
#'
#'       Default value:
#'
#'       Depends on object count and feature count in train dataset and on learning mode.
#'
#'   \item dev_score_calc_obj_block_size
#'
#'       CPU only. Size of block of samples in score calculation. Should be > 0
#'       Used only for learning speed tuning.
#'       Changing this parameter can affect results in pairwise scoring mode due to numerical accuracy differences
#'
#'       Default value:
#'
#'       5000000
#'
#'   \item dev_efb_max_buckets
#'
#'       CPU only. Maximum bucket count in exclusive features bundle. Should be in an integer between 0 and 65536.
#'       Used only for learning speed tuning.
#'
#'       Default value:
#'
#'       1024
#'
#'   \item sparse_features_conflict_fraction
#'
#'      CPU only. Maximum allowed fraction of conflicting non-default values for features in exclusive features bundle.
#'      Should be a real value in [0, 1) interval.
#'
#'      Default value:
#'
#'      0.0
#'
#'    \item leaf_estimation_backtracking
#'
#'        Type of backtracking during gradient descent.
#'        Possible values:
#'            - 'No' - never backtrack; supported on CPU and GPU
#'            - 'AnyImprovement' - reduce the descent step until the value of loss function is less than before the step; supported on CPU and GPU
#'            - 'Armijo' - reduce the descent step until Armijo condition is satisfied; supported on GPU only
#'
#'        Default value:
#'
#'        'AnyImprovement'
#'
#'   }
#' }
#'
#' % BEGIN GENERATED PARAM REFERENCE (tools/parity/gen_param_reference.py) -- DO NOT EDIT BY HAND
#' @section Full Parameter Reference (generated):
#' All 139 hyperparameters from the machine-generated capability
#' inventory, and how catboostr accepts each one.
#'
#' Unrecognized \code{params} keys are rejected with an error naming the
#' offending key(s). To use a newer vendor-core parameter not yet listed
#' below, set \code{options(catboostr.allow_unknown_params = TRUE)} to bypass
#' this check.
#' \describe{
#'   \item{X}{Supplied via \code{learn_pool}/\code{test_pool} (see \code{catboost.load_pool}), not the \code{params} list.}
#'   \item{allow_const_label}{Native training parameter accepted in the \code{params} list.}
#'   \item{allow_writing_files}{Native training parameter accepted in the \code{params} list.}
#'   \item{approx_on_full_history}{Native training parameter accepted in the \code{params} list.}
#'   \item{auto_class_weights}{Native training parameter accepted in the \code{params} list.}
#'   \item{bagging_temperature}{Native training parameter accepted in the \code{params} list.}
#'   \item{baseline}{Supplied via \code{catboost.load_pool} (Pool construction), not the \code{params} list.}
#'   \item{best_model_min_trees}{Native training parameter accepted in the \code{params} list.}
#'   \item{boost_from_average}{Native training parameter accepted in the \code{params} list.}
#'   \item{boosting_type}{Native training parameter accepted in the \code{params} list.}
#'   \item{bootstrap_type}{Native training parameter accepted in the \code{params} list.}
#'   \item{border_count}{Native training parameter accepted in the \code{params} list.}
#'   \item{callback}{Python-only; no catboostr equivalent.}
#'   \item{callbacks}{Python-only; no catboostr equivalent.}
#'   \item{cat_features}{Supplied via \code{catboost.load_pool} (Pool construction), not the \code{params} list.}
#'   \item{class_names}{Native training parameter accepted in the \code{params} list.}
#'   \item{class_weights}{Native training parameter accepted in the \code{params} list.}
#'   \item{classes_count}{Native training parameter accepted in the \code{params} list.}
#'   \item{colsample_bylevel}{Alias of \code{rsm} (see process_synonyms); resolved automatically.}
#'   \item{column_description}{Supplied via \code{catboost.load_pool} (Pool construction), not the \code{params} list.}
#'   \item{combinations_ctr}{Native training parameter accepted in the \code{params} list.}
#'   \item{counter_calc_method}{Native training parameter accepted in the \code{params} list.}
#'   \item{ctr_description}{Native training parameter accepted in the \code{params} list.}
#'   \item{ctr_history_unit}{Native training parameter accepted in the \code{params} list.}
#'   \item{ctr_leaf_count_limit}{Native training parameter accepted in the \code{params} list.}
#'   \item{ctr_target_border_count}{Native training parameter accepted in the \code{params} list.}
#'   \item{custom_loss}{Native training parameter accepted in the \code{params} list.}
#'   \item{custom_metric}{Native training parameter accepted in the \code{params} list.}
#'   \item{data_partition}{Native training parameter accepted in the \code{params} list.}
#'   \item{depth}{Native training parameter accepted in the \code{params} list.}
#'   \item{dev_efb_max_buckets}{Native training parameter accepted in the \code{params} list.}
#'   \item{dev_score_calc_obj_block_size}{Native training parameter accepted in the \code{params} list.}
#'   \item{device_config}{Native training parameter accepted in the \code{params} list.}
#'   \item{devices}{Native training parameter accepted in the \code{params} list.}
#'   \item{dictionaries}{Native training parameter accepted in the \code{params} list.}
#'   \item{diffusion_temperature}{Native training parameter accepted in the \code{params} list.}
#'   \item{early_stopping_rounds}{Native training parameter accepted in the \code{params} list.}
#'   \item{embedding_features}{Supplied via \code{catboost.load_pool} (Pool construction), not the \code{params} list.}
#'   \item{eta}{Alias of \code{learning_rate} (see process_synonyms); resolved automatically.}
#'   \item{eval_fraction}{Native training parameter accepted in the \code{params} list.}
#'   \item{eval_metric}{Native training parameter accepted in the \code{params} list.}
#'   \item{eval_set}{Supplied via the \code{test_pool} argument of \code{catboost.train}, not the \code{params} list.}
#'   \item{feature_border_type}{Native training parameter accepted in the \code{params} list.}
#'   \item{feature_calcers}{Native training parameter accepted in the \code{params} list.}
#'   \item{feature_weights}{Native training parameter accepted in the \code{params} list.}
#'   \item{final_ctr_computation_mode}{Native training parameter accepted in the \code{params} list.}
#'   \item{first_feature_use_penalties}{Native training parameter accepted in the \code{params} list.}
#'   \item{fixed_binary_splits}{Native training parameter accepted in the \code{params} list.}
#'   \item{fold_len_multiplier}{Native training parameter accepted in the \code{params} list.}
#'   \item{fold_permutation_block}{Native training parameter accepted in the \code{params} list.}
#'   \item{gpu_cat_features_storage}{Native training parameter accepted in the \code{params} list.}
#'   \item{gpu_ram_part}{Native training parameter accepted in the \code{params} list.}
#'   \item{graph}{Supplied via \code{catboost.load_pool} (Pool construction), not the \code{params} list.}
#'   \item{group_id}{Supplied via \code{catboost.load_pool} (Pool construction), not the \code{params} list.}
#'   \item{group_weight}{Supplied via \code{catboost.load_pool} (Pool construction), not the \code{params} list.}
#'   \item{grow_policy}{Native training parameter accepted in the \code{params} list.}
#'   \item{has_time}{Native training parameter accepted in the \code{params} list.}
#'   \item{ignored_features}{Native training parameter accepted in the \code{params} list.}
#'   \item{init_model}{Supplied via the \code{init_model} argument of \code{catboost.train}, not the \code{params} list.}
#'   \item{input_borders}{Native training parameter accepted in the \code{params} list.}
#'   \item{iterations}{Native training parameter accepted in the \code{params} list.}
#'   \item{l2_leaf_reg}{Native training parameter accepted in the \code{params} list.}
#'   \item{langevin}{Native training parameter accepted in the \code{params} list.}
#'   \item{leaf_estimation_backtracking}{Native training parameter accepted in the \code{params} list.}
#'   \item{leaf_estimation_iterations}{Native training parameter accepted in the \code{params} list.}
#'   \item{leaf_estimation_method}{Native training parameter accepted in the \code{params} list.}
#'   \item{learning_rate}{Native training parameter accepted in the \code{params} list.}
#'   \item{log_cerr}{Python-only; no catboostr equivalent.}
#'   \item{log_cout}{Python-only; no catboostr equivalent.}
#'   \item{logging_level}{Native training parameter accepted in the \code{params} list.}
#'   \item{loss_function}{Native training parameter accepted in the \code{params} list.}
#'   \item{max_bin}{Alias of \code{border_count} (see process_synonyms); resolved automatically.}
#'   \item{max_ctr_complexity}{Native training parameter accepted in the \code{params} list.}
#'   \item{max_depth}{Alias of \code{depth} (see process_synonyms); resolved automatically.}
#'   \item{max_leaves}{Native training parameter accepted in the \code{params} list.}
#'   \item{metadata}{Native training parameter accepted in the \code{params} list.}
#'   \item{metric_period}{Native training parameter accepted in the \code{params} list.}
#'   \item{min_child_samples}{Alias of \code{min_data_in_leaf} (see process_synonyms); resolved automatically.}
#'   \item{min_data_in_leaf}{Native training parameter accepted in the \code{params} list.}
#'   \item{model_shrink_mode}{Native training parameter accepted in the \code{params} list.}
#'   \item{model_shrink_rate}{Native training parameter accepted in the \code{params} list.}
#'   \item{model_size_reg}{Native training parameter accepted in the \code{params} list.}
#'   \item{monotone_constraints}{Native training parameter accepted in the \code{params} list.}
#'   \item{mvs_reg}{Native training parameter accepted in the \code{params} list.}
#'   \item{n_estimators}{Alias of \code{iterations} (see process_synonyms); resolved automatically.}
#'   \item{name}{Native training parameter accepted in the \code{params} list.}
#'   \item{nan_mode}{Native training parameter accepted in the \code{params} list.}
#'   \item{num_boost_round}{Alias of \code{iterations} (see process_synonyms); resolved automatically.}
#'   \item{num_leaves}{Alias of \code{max_leaves} (see process_synonyms); resolved automatically.}
#'   \item{num_trees}{Alias of \code{iterations} (see process_synonyms); resolved automatically.}
#'   \item{objective}{Alias of \code{loss_function} (see process_synonyms); resolved automatically.}
#'   \item{od_pval}{Native training parameter accepted in the \code{params} list.}
#'   \item{od_type}{Native training parameter accepted in the \code{params} list.}
#'   \item{od_wait}{Native training parameter accepted in the \code{params} list.}
#'   \item{one_hot_max_size}{Native training parameter accepted in the \code{params} list.}
#'   \item{output_borders}{Native training parameter accepted in the \code{params} list.}
#'   \item{pairs}{Supplied via \code{catboost.load_pool} (Pool construction), not the \code{params} list.}
#'   \item{pairs_weight}{Supplied via \code{catboost.load_pool} (Pool construction), not the \code{params} list.}
#'   \item{params}{Refers to the \code{params} argument itself.}
#'   \item{penalties_coefficient}{Native training parameter accepted in the \code{params} list.}
#'   \item{per_feature_ctr}{Native training parameter accepted in the \code{params} list.}
#'   \item{per_float_feature_quantization}{Native training parameter accepted in the \code{params} list.}
#'   \item{per_object_feature_penalties}{Native training parameter accepted in the \code{params} list.}
#'   \item{pinned_memory_size}{Native training parameter accepted in the \code{params} list.}
#'   \item{plot}{Python-only; no catboostr equivalent.}
#'   \item{plot_file}{Python-only; no catboostr equivalent.}
#'   \item{posterior_sampling}{Native training parameter accepted in the \code{params} list.}
#'   \item{random_score_type}{Native training parameter accepted in the \code{params} list.}
#'   \item{random_seed}{Native training parameter accepted in the \code{params} list.}
#'   \item{random_state}{Alias of \code{random_seed} (see process_synonyms); resolved automatically.}
#'   \item{random_strength}{Native training parameter accepted in the \code{params} list.}
#'   \item{reg_lambda}{Alias of \code{l2_leaf_reg} (see process_synonyms); resolved automatically.}
#'   \item{rsm}{Native training parameter accepted in the \code{params} list.}
#'   \item{sample_weight}{Supplied via \code{catboost.load_pool} (Pool construction), not the \code{params} list.}
#'   \item{sampling_frequency}{Native training parameter accepted in the \code{params} list.}
#'   \item{sampling_unit}{Native training parameter accepted in the \code{params} list.}
#'   \item{save_snapshot}{Native training parameter accepted in the \code{params} list.}
#'   \item{scale_pos_weight}{Native training parameter accepted in the \code{params} list.}
#'   \item{score_function}{Native training parameter accepted in the \code{params} list.}
#'   \item{silent}{Python-only; no catboostr equivalent.}
#'   \item{simple_ctr}{Native training parameter accepted in the \code{params} list.}
#'   \item{snapshot_file}{Native training parameter accepted in the \code{params} list.}
#'   \item{snapshot_interval}{Native training parameter accepted in the \code{params} list.}
#'   \item{sparse_features_conflict_fraction}{Native training parameter accepted in the \code{params} list.}
#'   \item{store_all_simple_ctr}{Native training parameter accepted in the \code{params} list.}
#'   \item{subgroup_id}{Supplied via \code{catboost.load_pool} (Pool construction), not the \code{params} list.}
#'   \item{subsample}{Native training parameter accepted in the \code{params} list.}
#'   \item{target_border}{Native training parameter accepted in the \code{params} list.}
#'   \item{task_type}{Native training parameter accepted in the \code{params} list.}
#'   \item{text_features}{Supplied via \code{catboost.load_pool} (Pool construction), not the \code{params} list.}
#'   \item{text_processing}{Native training parameter accepted in the \code{params} list.}
#'   \item{thread_count}{Supplied via \code{catboost.load_pool} (Pool construction), not the \code{params} list.}
#'   \item{tokenizers}{Native training parameter accepted in the \code{params} list.}
#'   \item{train_dir}{Native training parameter accepted in the \code{params} list.}
#'   \item{use_best_model}{Native training parameter accepted in the \code{params} list.}
#'   \item{used_ram_limit}{Native training parameter accepted in the \code{params} list.}
#'   \item{verbose}{Native training parameter accepted in the \code{params} list.}
#'   \item{verbose_eval}{Alias of \code{verbose} (see process_synonyms); resolved automatically.}
#'   \item{y}{Supplied via \code{learn_pool}/\code{test_pool} (see \code{catboost.load_pool}), not the \code{params} list.}
#' }
#' % END GENERATED PARAM REFERENCE
#'
#' @details \strong{Custom objective / custom eval metric support matrix.}
#' Every entry point below shares the same
#' \code{custom_objective}/\code{custom_eval_metric_object} argument contract
#' documented under \code{catboost.train}'s own \code{custom_objective} and
#' \code{custom_eval_metric_object} params:
#' \itemize{
#'   \item{\code{catboost.train}}{ Both \code{custom_objective} and
#'     \code{custom_eval_metric_object}.}
#'   \item{\code{catboost.cv}}{ Both.}
#'   \item{\code{catboost.grid_search}}{ Both -- forwarded to the search
#'     itself and, when \code{refit = TRUE}, to the refit.}
#'   \item{\code{catboost.randomized_search}}{ Both -- same forwarding as
#'     \code{catboost.grid_search}.}
#'   \item{\code{catboost.eval_feature}}{ Both.}
#'   \item{\code{catboost.select_features}}{ \code{custom_eval_metric_object}
#'     \strong{only}. There is no \code{custom_objective} argument: the
#'     native entry point it calls, \code{NCB::SelectFeatures} (vendored at
#'     \code{catboost/libs/features_selection/select_features.h}), takes no
#'     \code{TCustomObjectiveDescriptor} parameter anywhere in its signature
#'     or in \code{recursive_features_elimination.*} -- there is no C++-side
#'     slot to wire a custom objective into, so the omission is structural,
#'     not a missing feature.}
#' }
#'
#' \strong{Multithreading preservation.} CatBoost's own engine parallelizes
#' derivative/metric computation across TBB worker threads
#' (\code{NPar::TTbbLocalExecutor}), but R's C API may only be called from R's
#' own main thread. The naive fix -- forcing \code{params$thread_count = 1}
#' whenever a custom objective/metric is supplied -- was rejected: it would
#' silently kill CatBoost's own training parallelism for every custom-loss
#' user, trading a correctness problem for a performance regression. Instead,
#' training runs on a background \code{std::thread} while R's real main
#' thread (the \code{.Call} frame the user invoked) runs a drain loop that
#' services R-closure-invocation requests queued by any worker thread; each
#' request blocks its own thread on a condition variable until the main
#' thread has executed it and posted the result back
#' (\code{src/r_callback_bridge.h}). TBB's own parallel scheduling of the
#' surrounding derivative/metric computation is untouched -- only the R
#' closure invocation itself is serialized onto the main thread, not the
#' whole computation. \code{params$thread_count} is therefore never forced
#' to 1 for a custom objective/metric run.
#'
#' \strong{Ctrl-C during a bridged run.} Detecting the
#' interrupt on the drain loop's polling thread consumes it -- R clears its
#' pending-interrupt flag before the drain loop observes it, so nothing is
#' left for R to deliver afterwards. Ctrl-C during a custom-objective/metric
#' run therefore surfaces as an ordinary R error, not an R interrupt
#' condition: \code{tryCatch(..., interrupt = )} will not fire on it.
#' Restoring true interrupt semantics is tracked as a follow-up, not pursued
#' here due to CRAN \code{R CMD check} NOTE risk in the required mechanism.
#'
#' \strong{Distributed training.} \code{params} accepts the native
#' \code{node_type}, \code{node_port} and \code{file_with_hosts} keys used to
#' run multi-machine training: start a worker process on each remote machine
#' with \code{\link{catboost.run_worker}}, then set \code{node_type =
#' "Master"}, \code{file_with_hosts = <path to a "host:port"-per-line file>}
#' and \code{node_port = <this machine's own par-framework port>} in the
#' master-side \code{params} list. Setting any \code{node_type} other than
#' \code{"SingleHost"} routes the call to the native distributed training
#' engine (the same one the command-line client's \code{--node-type Master}
#' fit uses); the default single-host path is untouched by this and behaves
#' exactly as before.
#'
#' Distributed training shards the dataset across the workers and builds each
#' split's histograms per shard, so a distributed model is not bit-for-bit
#' identical to a single-host model trained on the same data with the same
#' seed -- it is statistically equivalent, not reproducible against it.
#' \code{node_type = "Master"} requires \code{task_type = "CPU"} (the
#' default); it is rejected for GPU training.
#'
#' @param learn_pool The dataset used for training the model.
#'
#' Default value: Required argument
#' @param test_pool The dataset used for testing the quality of the model.
#'
#' Default value: NULL (not used)
#' @param params The list of parameters to start training with.
#'
#' If omitted, default values are used (see The list of parameters).
#'
#' If set, the passed list of parameters overrides the default values.
#'
#' Default value: Required argument
#' @param init_model Continue training starting from an existing model.
#'
#' Accepts a \code{catboost.Model} object (as returned by \code{catboost.train}
#' or \code{catboost.load_model}), or a string/path to a model file on disk
#' (loaded via \code{catboost.load_model} with the default \code{"cbm"}
#' format).
#'
#' Default value: NULL (train a new model from scratch)
#' @param custom_objective A user-defined loss function, used instead of
#' \code{params$loss_function}. Must be a named \code{list} with one or both
#' of the following elements (both are R closures; at least one is
#' required):
#' \itemize{
#'   \item{\code{calc_ders_range = function(approx, target, weight)}}{
#'     Called for single-dimension losses (regression, binary
#'     classification, ranking). \code{approx}/\code{target} are numeric
#'     vectors of the current batch's predictions/labels; \code{weight} is
#'     either a numeric vector of the same length or \code{NULL} (unweighted
#'     pool). Must return an \code{N x 2} numeric matrix (\code{N} =
#'     \code{length(approx)}), column 1 = first derivative, column 2 =
#'     second derivative of the loss w.r.t. \code{approx}, one row per
#'     observation.}
#'   \item{\code{calc_ders_multi = function(approx, target, weight)}}{
#'     Called for multi-dimensional losses (multiclass, multi-target
#'     regression), once per observation. \code{approx} is a numeric vector
#'     of length \code{K} (the current observation's per-dimension
#'     predictions); \code{target} is a numeric vector (length 1 for
#'     multiclass, length \code{K} for multi-target); \code{weight} is a
#'     single number. Must return \code{list(der1 = <numeric vector, length
#'     K>, der2 = <K x K numeric matrix, or NULL>)}: \code{der1} is the
#'     gradient: \code{der2}, when required by the current
#'     \code{leaf_estimation_method}, is the Hessian (a symmetric matrix --
#'     only its upper triangle, including the diagonal, is read).}
#' }
#' Reimplements the two-method contract of Python's
#' \code{CatBoost(loss_function = <object with calc_ders_range/
#' calc_ders_multi>)} (see \code{_catboost.pyx}'s
#' \code{_ObjectiveCalcDersRange}/\code{_ObjectiveCalcDersMultiClass}/
#' \code{_ObjectiveCalcDersMultiTarget}), adapted to R's lack of a
#' scalar/length-1-vector distinction. Every call is marshaled back onto R's
#' main thread (via the package's internal callback bridge) so it is safe to
#' call from a multi-threaded training run; \code{params$thread_count} is NOT
#' forced to 1.
#'
#' \code{params$loss_function} is set automatically to
#' \code{"PythonUserDefinedPerObject"} (covers \code{calc_ders_range} and
#' \code{calc_ders_multi} used for MultiClass-shaped losses) when left
#' unset; set it explicitly to \code{"PythonUserDefinedMultiTarget"}
#' yourself if \code{calc_ders_multi} implements a multi-target regression
#' objective (vector target) instead -- any other value is rejected. A
#' custom objective also requires \code{params$eval_metric} to be set
#' explicitly (there is no default metric to infer from an opaque R
#' closure), and is incompatible with \code{params$boost_from_average =
#' TRUE} (native CatBoost only computes that data-dependent starting bias
#' for a fixed allowlist of built-in losses).
#'
#' Default value: NULL (use \code{params$loss_function} as-is)
#' @param custom_eval_metric_object A user-defined eval metric, used instead
#' of \code{params$eval_metric}. Must be a named \code{list} with:
#' \itemize{
#'   \item{\code{evaluate = function(approx, target, weight)}}{ Required.
#'     Called once per batch. \code{approx} is a numeric \code{N x D} matrix
#'     (\code{N} = batch size, \code{D} = number of approx dimensions --
#'     \code{D == 1} for single-dimension losses). \code{target} is a numeric
#'     vector of length \code{N} (single-target metrics, \code{multi_target}
#'     unset/\code{FALSE}) or a numeric \code{N x K} matrix (multi-target
#'     metrics, \code{multi_target = TRUE}). \code{weight} is a numeric
#'     vector of length \code{N}, or \code{NULL} (unweighted pool). Must
#'     return \code{list(error = <numeric scalar>, weight = <numeric
#'     scalar>)}: the batch's weighted error sum and weight sum -- CatBoost
#'     itself sums these across batches/folds.}
#'   \item{\code{is_max_optimal = function()}}{ Required. Returns
#'     \code{TRUE}/\code{FALSE}: whether a larger metric value is better.}
#'   \item{\code{get_final_error = function(error)}}{ Optional.
#'     \code{error} is the numeric length-2 vector \code{c(sum_error,
#'     sum_weight)} accumulated across all batches (the same two numbers
#'     \code{evaluate} reported, summed). Must return the final scalar
#'     metric value. Defaults to \code{sum_error / sum_weight} (or \code{0}
#'     if \code{sum_weight == 0}) when omitted, matching native
#'     \code{IMetric}'s own default (see
#'     \code{vendor/catboost/catboost/libs/metrics/metric.cpp}'s
#'     \code{TMetric::GetFinalError}).}
#'   \item{\code{is_additive = function()}}{ Optional. Returns
#'     \code{TRUE}/\code{FALSE}: whether per-batch \code{error}/\code{weight}
#'     sums can be combined by plain addition. Defaults to \code{FALSE} when
#'     omitted (conservative; matches Python's own default).}
#'   \item{\code{multi_target}}{ Optional \code{logical} scalar, default
#'     \code{FALSE}. Set \code{TRUE} if \code{evaluate}'s \code{target}
#'     argument is a multi-column matrix (a multi-target regression metric)
#'     rather than a single vector.}
#' }
#' Reimplements the four-method contract of Python's
#' \code{CustomMetric}/\code{MultiTargetCustomMetric}
#' (\code{_catboost.pyx}'s \code{evaluate}/\code{is_max_optimal}/
#' \code{get_final_error}/\code{is_additive}), adapted to R's lack of a class
#' hierarchy: Python dispatches single- vs multi-target \code{evaluate} by
#' subclassing \code{MultiTargetCustomMetric}; R uses the explicit
#' \code{multi_target} list element instead. Every call is marshaled back
#' onto R's main thread via the same callback bridge \code{custom_objective}
#' uses (shared, not a second queue); \code{params$thread_count} is NOT
#' forced to 1.
#'
#' \code{params$eval_metric} is set automatically to
#' \code{"PythonUserDefinedPerObject"} (\code{multi_target} unset/
#' \code{FALSE}) or \code{"PythonUserDefinedMultiTarget"} (\code{multi_target
#' = TRUE}) when left unset; any other value is rejected. This argument is a
#' deliberate R-vs-Python API divergence: Python passes a metric object
#' directly through \code{eval_metric = <object>}, while R keeps
#' \code{eval_metric} as its existing native string/params-list key (see
#' \code{custom_metric}, a \emph{different}, pre-existing native parameter
#' for reporting extra built-in metrics -- not a callback) and adds this
#' separate formal argument instead, to avoid a naming collision between the
#' two.
#'
#' Default value: NULL (use \code{params$eval_metric} as-is)
#' @examples
#' \dontrun{
#' train_pool_path <- system.file("extdata", "adult_train.1000", package = "catboostr")
#' test_pool_path <- system.file("extdata", "adult_test.1000", package = "catboostr")
#' cd_path <- system.file("extdata", "adult.cd", package = "catboostr")
#' train_pool <- catboost.load_pool(train_pool_path, column_description = cd_path)
#' test_pool <- catboost.load_pool(test_pool_path, column_description = cd_path)
#' fit_params <- list(
#'     iterations = 100,
#'     loss_function = 'Logloss',
#'     ignored_features = c(4, 9),
#'     border_count = 32,
#'     depth = 5,
#'     learning_rate = 0.03,
#'     l2_leaf_reg = 3.5,
#'     train_dir = 'train_dir')
#' model <- catboost.train(train_pool, test_pool, fit_params)
#' }
#'
#' # (a) custom_objective: reimplement RMSE's own derivatives (matches
#' # calc_ders_range's contract -- weight is applied by the closure itself,
#' # unlike the built-in dispatcher, which multiplies by weight for you).
#' \dontrun{
#' n <- 50
#' features <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
#' label <- with(features, x1 - x2 + rnorm(n, sd = 0.3))
#' pool <- catboost.load_pool(features, label = label)
#' rmse_custom_objective <- list(
#'     calc_ders_range = function(approx, target, weight) {
#'         w <- if (is.null(weight)) 1 else weight
#'         cbind(w * (target - approx), w * (-1))
#'     })
#' model <- catboost.train(
#'     pool,
#'     params = list(iterations = 10, depth = 3, logging_level = "Silent",
#'                   eval_metric = "RMSE", boost_from_average = FALSE),
#'     custom_objective = rmse_custom_objective)
#' }
#'
#' # (b) custom_eval_metric_object: reimplement RMSE's own Eval/GetFinalError.
#' \dontrun{
#' n <- 50
#' features <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
#' label <- with(features, x1 - x2 + rnorm(n, sd = 0.3))
#' pool <- catboost.load_pool(features, label = label)
#' rmse_custom_eval_metric <- list(
#'     evaluate = function(approx, target, weight) {
#'         w <- if (is.null(weight)) rep(1, length(target)) else weight
#'         diff <- approx[, 1] - target
#'         list(error = sum(w * diff^2), weight = sum(w))
#'     },
#'     is_max_optimal = function() FALSE,
#'     get_final_error = function(error) sqrt(error[1] / (error[2] + 1e-38)))
#' model <- catboost.train(
#'     pool,
#'     params = list(iterations = 10, depth = 3, logging_level = "Silent"),
#'     custom_eval_metric_object = rmse_custom_eval_metric)
#' }
#' @return Model object.
#' @export
#' @seealso \url{https://catboost.ai/docs/concepts/r-reference_catboost-train.html}
catboost.train <- function(learn_pool, test_pool = NULL, params = list(), init_model = NULL, custom_objective = NULL, custom_eval_metric_object = NULL) {
    if (!inherits(learn_pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(learn_pool))
    if (is.null.handle(learn_pool))
        stop("'learn_pool' object is invalid.")
    if (!is.null(test_pool) && !inherits(test_pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(test_pool))
    if (!is.null(test_pool) && is.null.handle(test_pool))
        stop("'test_pool' object is invalid.")
    if (length(params) == 0)
        message("Training catboost with default parameters! See help(catboost.train).")

    # P5.1 (catboost-8z4.58): init_model matches Python's CatBoost.fit(init_model=...),
    # which accepts a CatBoost object or a string/path (fit() loads the path
    # via CatBoost().load_model() before passing it to _train()).
    init_model_handle <- NULL
    if (!is.null(init_model)) {
        if (is.character(init_model))
            init_model <- catboost.load_model(init_model)
        if (!inherits(init_model, "catboost.Model"))
            stop("Expected catboost.Model or a path to a model file, got: ", class(init_model))
        catboost.restore_handle(init_model)
        init_model_handle <- init_model$cpp_obj$handle
    }

    # P6.3/P6.4 (catboost-8z4.89/.90): custom_objective/custom_eval_metric_object
    # travel as their own .Call arguments -- params is jsonlite::toJSON'd below
    # and has no asJSON method for R closures, so an R function/list cannot
    # cross that path. Validation/defaulting factored into shared helpers
    # (catboost-8z4.94) now that 5 more entry points need the identical checks.
    params <- process_synonyms(params)
    params <- apply_custom_objective_params(params, custom_objective)
    params <- apply_custom_eval_metric_params(params, custom_eval_metric_object)

    json_params <- prepare_train_export_parameters(params)
    handle <- .Call("CatBoostFit_R", learn_pool, test_pool, json_params, init_model_handle, custom_objective, custom_eval_metric_object)
    raw <- .Call("CatBoostSerializeModel_R", handle)
    model <- create.model.base(handle, raw)

    if (catboost._is_oblivious(model)) {
        if (catboost._is_groupwise_metric(model)) {
            # too expensive
        } else {
            model$feature_importances <- catboost.get_feature_importance(model)
            rownames(model$feature_importances) <- colnames(learn_pool)
        }
    }

    plain_params <- catboost.get_plain_params(model)
    model$tree_count <- catboost.ntrees(model)
    model$learning_rate <- plain_params$learning_rate
    model$feature_count <- ncol(learn_pool) - length(plain_params$ignored_features)
    return(model)
}

# BEGIN GENERATED KNOWN_PARAMS (tools/parity/gen_param_reference.py) -- DO NOT EDIT BY HAND
# Canonical + alias hyperparameter names from the machine-generated
# capability inventory (tests/fixtures/parity/matrix.dispositioned.json,
# kind:"parameter" rows), unioned with every name vendor-native's
# CopyOption(plainOptions, ...) calls accept (plain_options_helper.cpp;
# see parse_native_copyoption_names() in this script), plus
# EXTRA_KNOWN_PARAMS. Used by validate_params_keys() below.
.catboostr_known_params <- c("X", "add_ridge_penalty_to_loss_function", 
    "allow_const_label", "allow_writing_files", "approx_on_full_history", 
    "auto_class_weights", "bagging_temperature", "baseline", "baseline_model_snapshot", 
    "bayesian_matrix_reg", "best_model_min_trees", "boost_from_average", "boosting_type", 
    "bootstrap_type", "border_count", "callback", "callbacks", "cat_features", 
    "class_names", "class_weights", "classes_count", "colsample_bylevel", 
    "column_description", "combinations_ctr", "counter_calc_method", "ctr_description", 
    "ctr_history_unit", "ctr_leaf_count_limit", "ctr_target_border_count", "custom_loss", 
    "custom_metric", "data_partition", "depth", "detailed_profile", 
    "dev_default_value_fraction_for_sparse", "dev_efb_max_buckets", "dev_group_features", 
    "dev_leafwise_approxes", "dev_leafwise_scoring", 
    "dev_max_ctr_complexity_for_borders_cache", "dev_max_subset_size_for_build_borders", 
    "dev_score_calc_obj_block_size", "dev_sparse_array_indexing", "device_config", 
    "devices", "dictionaries", "diffusion_temperature", "early_stopping_rounds", 
    "embedding_features", "embedding_processing", "eta", "eval_file_name", 
    "eval_fraction", "eval_metric", "eval_set", "experiment_count", "experiment_size", 
    "feature_border_type", "feature_calcers", "feature_weights", "features_for_select", 
    "features_selection_algorithm", "features_selection_grouping", 
    "features_selection_result_path", "features_selection_steps", 
    "features_tags_for_select", "features_to_evaluate", "file_with_hosts", 
    "final_ctr_computation_mode", "final_feature_calcer_computation_mode", 
    "first_feature_use_penalties", "fixed_binary_splits", "fold_len_multiplier", 
    "fold_permutation_block", "fold_size_loss_normalization", 
    "force_unit_auto_pair_weights", "fstr_internal_file", "fstr_regular_file", 
    "fstr_type", "gpu_cat_features_storage", "gpu_ram_part", "graph", "group_id", 
    "group_weight", "grow_policy", "has_time", "ignored_features", "init_model", 
    "input_borders", "iterations", "json_log", "l2_leaf_reg", "langevin", 
    "leaf_estimation_backtracking", "leaf_estimation_iterations", 
    "leaf_estimation_method", "learn_error_log", "learning_rate", "log_cerr", "log_cout", 
    "logging_level", "loss_function", "max_bin", "max_ctr_complexity", "max_depth", 
    "max_leaves", "meta", "meta_l2_exponent", "meta_l2_frequency", "metadata", 
    "metric_period", "min_child_samples", "min_data_in_leaf", "min_fold_size", 
    "model_format", "model_shrink_mode", "model_shrink_rate", "model_size_reg", 
    "monotone_constraints", "mvs_reg", "n_estimators", "name", "nan_mode", "node_port", 
    "node_type", "num_boost_round", "num_features_tags_to_select", 
    "num_features_to_select", "num_leaves", "num_trees", "objective", 
    "observations_to_bootstrap", "od_pval", "od_type", "od_wait", "offset", 
    "one_hot_max_size", "output_borders", "output_columns", "pairs", "pairs_weight", 
    "params", "penalties_coefficient", "per_feature_ctr", "per_feature_ctr_description", 
    "per_float_feature_quantization", "per_object_feature_penalties", 
    "permutation_count", "pinned_memory_size", "plot", "plot_file", 
    "pool_metainfo_options", "posterior_sampling", "prediction_type", "profile_log", 
    "random_score_type", "random_seed", "random_state", "random_strength", "reg_lambda", 
    "result_model_file", "roc_file", "rsm", "sample_weight", "sampling_frequency", 
    "sampling_unit", "save_snapshot", "scale_pos_weight", "score_function", 
    "shap_calc_type", "silent", "simple_ctr", "simple_ctr_description", "snapshot_file", 
    "snapshot_interval", "sparse_features_conflict_fraction", "store_all_simple_ctr", 
    "subgroup_id", "subsample", "target_border", "task_type", "test_error_log", 
    "text_features", "text_processing", "thread_count", "time_left_log", "tokenizers", 
    "train_dir", "train_final_model", "training_options_file", "tree_ctr_description", 
    "use_best_model", "use_evaluated_features_in_baseline_model", "used_ram_limit", 
    "verbose", "verbose_eval", "y")
# END GENERATED KNOWN_PARAMS

# P5.5 (catboost-8z4.62): reject unknown `params` list keys, unless the
# caller opts out via options(catboostr.allow_unknown_params = TRUE) (e.g.
# to use a newer vendor-core parameter not yet in .catboostr_known_params).
# Must run *after* synonym resolution: aliases (e.g. 'eta') are themselves
# members of .catboostr_known_params, so checking post-resolution or
# pre-resolution both accept them, but post-resolution also means the
# canonical spelling is what actually gets validated and serialized.
# Duplicate-key check factored out (catboost-8z4.95 batch review finding):
# validate_params_keys() alone is not enough for catboost.train/grid_search/
# randomized_search/select_features, which all call process_synonyms() first.
# process_synonyms_in_one_group()'s `params[[synonym]] <- NULL` removes only
# the FIRST match of a duplicated alias-group name (e.g. two 'iterations'
# entries from c(list(iterations=5), list(iterations=9))), silently keeping
# the first value and dropping the second BEFORE validate_params_keys ever
# runs -- the opposite of erroring, and opposite of what a caller overriding
# via c() expects. Must run on the RAW params, before any synonym resolution.
check_no_duplicate_params_keys <- function(params) {
    dup <- unique(names(params)[duplicated(names(params))])
    if (length(dup) > 0) {
        stop("Duplicate 'params' key(s): ", paste(dup, collapse = ", "),
             ". jsonlite::toJSON() silently renames repeated keys (e.g. 'a.1'), ",
             "which native CatBoost then rejects or misapplies. Use modifyList() ",
             "or list assignment to override a key instead of concatenating lists with c().")
    }
    invisible(NULL)
}

validate_params_keys <- function(params) {
    check_no_duplicate_params_keys(params)
    if (isTRUE(getOption("catboostr.allow_unknown_params", FALSE))) {
        return(invisible(NULL))
    }
    unknown <- setdiff(names(params), .catboostr_known_params)
    if (length(unknown) > 0) {
        stop("Unknown catboost 'params' key(s): ", paste(unknown, collapse = ", "),
             ". See help(catboost.train) for the full parameter reference. ",
             "If this is a newer vendor-core parameter not yet recognized by catboostr, ",
             "set options(catboostr.allow_unknown_params = TRUE) to bypass this check.")
    }
    invisible(NULL)
}

process_synonyms <- function(params) {
    # Must check for raw duplicate keys BEFORE any alias-group resolution
    # below: process_synonyms_in_one_group() silently drops all but the
    # first occurrence of a duplicated alias-group name (e.g. two
    # 'iterations' entries), which would otherwise defeat this check by the
    # time validate_params_keys() runs at the end of this function.
    check_no_duplicate_params_keys(params)

    params <- process_synonyms_in_one_group(c('loss_function', 'objective'), params)
    params <- process_synonyms_in_one_group(c('iterations', 'n_estimators', 'num_boost_round', 'num_trees'), params)
    params <- process_synonyms_in_one_group(c('learning_rate', 'eta'), params)
    params <- process_synonyms_in_one_group(c('random_seed', 'random_state'), params)
    params <- process_synonyms_in_one_group(c('l2_leaf_reg', 'reg_lambda'), params)
    params <- process_synonyms_in_one_group(c('depth', 'max_depth'), params)
    params <- process_synonyms_in_one_group(c('min_data_in_leaf', 'min_child_samples'), params)
    params <- process_synonyms_in_one_group(c('max_leaves', 'num_leaves'), params)
    params <- process_synonyms_in_one_group(c('rsm', 'colsample_bylevel'), params)
    params <- process_synonyms_in_one_group(c('border_count', 'max_bin'), params)
    params <- process_synonyms_in_one_group(c('verbose', 'verbose_eval'), params)

    validate_params_keys(params)

    return(params)
}

process_synonyms_in_one_group <- function(synonyms, params) {
    value <- NULL
    for (synonym in synonyms) {
        if (synonym %in% names(params)) {
            if (!is.null(value)) {
                message <- paste("Only one of the parameters [",
                                paste(synonyms, collapse=', '),
                                "] should be initialized.", sep="")
                stop(message)
            }
            value = params[[synonym]]
            params[[synonym]] <- NULL
        }
   }

  if (!is.null(value)) {
      params[[synonyms[[1]]]] <- value
  }

  return(params)
}

prepare_train_export_parameters <- function(params) {

    if (length(params) == 0) {
        return ("{}")
    }

    if (!is.null(params$early_stopping_rounds)) {
        params$od_type <- "Iter"
        params$od_pval <- NULL
        params$od_wait <- params$early_stopping_rounds
        params$early_stopping_rounds <- NULL
    }

    if (!is.null(params$per_float_feature_quantization)) {
        params$per_float_feature_quantization <- as.list(params$per_float_feature_quantization)
    }

    if (!is.null(params$ignored_features)) {
        # treat parameter AsIs to avoid conversion from 1-element array to atomic value
        params$ignored_features <- I(as.character(params$ignored_features))
    }

    # digits = NA: jsonlite's full-round-trip-precision mode. digits = 10
    # (jsonlite's default) truncated hyperparameter values (e.g. learning_rate,
    # l2_leaf_reg) to 10 significant digits before they ever reached the
    # native training call -- same defect class as prepare_grid_json's fix
    # below (catboost-8z4.60), but here at the shared catboost.train/catboost.cv
    # serialization site (catboost-8z4.65).
    return(jsonlite::toJSON(params, auto_unbox = TRUE, digits = NA))
}

#' @name catboost.cv
#' @title Cross-validate model.
#' @description Estimate model performance using cross-validation.
#' @param pool Data to cross-validate on
#' @param params Parameters for catboost.train
#' @param fold_count Folds count.
#' @param type is type of cross-validation.
#' @param partition_random_seed The random seed used for splitting pool into folds.
#' @param shuffle Shuffle the dataset objects before splitting into folds.
#' @param stratified Perform stratified sampling.
#' @param early_stopping_rounds Activates Iter overfitting detector with od_wait set to early_stopping_rounds.
#' @param custom_objective A user-defined loss function, used instead of
#' \code{params$loss_function}. See \code{\link{catboost.train}}'s
#' \code{custom_objective} argument for the full contract.
#'
#' Default value: NULL (use \code{params$loss_function} as-is)
#' @param custom_eval_metric_object A user-defined eval metric, used instead
#' of \code{params$eval_metric}. See \code{\link{catboost.train}}'s
#' \code{custom_eval_metric_object} argument for the full contract.
#'
#' Default value: NULL (use \code{params$eval_metric} as-is)
#' @return A data.frame of evaluation results from cross-validation.
#' @export
catboost.cv <- function(pool,
                        params = list(),
                        fold_count = 3,
                        type = "Classical",
                        partition_random_seed = 0,
                        shuffle = TRUE,
                        stratified = FALSE,
                        early_stopping_rounds = NULL,
                        custom_objective = NULL,
                        custom_eval_metric_object = NULL) {

    if (!inherits(pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(pool))
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    if (length(params) == 0)
        message("Training catboost with default parameters! See help(catboost.train).")

    if (!is.null(early_stopping_rounds)) {
        params$od_type <- "Iter"
        params$od_pval <- NULL
        params$od_wait <- early_stopping_rounds
    }

    # P6.5 (catboost-8z4.94): same validation/defaulting as catboost.train,
    # via the shared helpers factored out there.
    params <- apply_custom_objective_params(params, custom_objective)
    params <- apply_custom_eval_metric_params(params, custom_eval_metric_object)

    # P5.5 (catboost-8z4.62): unlike catboost.train/grid_search/randomized_search,
    # catboost.cv does not call process_synonyms() (pre-existing behavior, left
    # untouched here), but every process_synonyms alias name is itself a member
    # of .catboostr_known_params, so validating directly against it still
    # accepts both canonical and alias spellings without requiring resolution.
    validate_params_keys(params)

    json_params <- prepare_train_export_parameters(params)
    result <- .Call("CatBoostCV_R", json_params, pool, fold_count, type, partition_random_seed, shuffle, stratified,
                     custom_objective, custom_eval_metric_object)

    return(data.frame(result))
}

#' @name catboost.run_worker
#' @title Run a distributed-training worker.
#' @description R equivalent of the CatBoost CLI's \code{run-worker} mode:
#' blocks the calling process, acting as a distributed-training worker
#' awaiting commands from a master node over the network, until stopped.
#' There is no R-level stop mechanism beyond what the CLI mode itself
#' provides -- killing the process, or the master sending a stop-slave
#' command through the par protocol.
#'
#' Call this from a separate R process (or session) on each worker machine,
#' \emph{before} starting the master-side training call. On the master side,
#' pass \code{params = list(node_type = "Master", file_with_hosts = <path>,
#' node_port = <port>)} to \code{\link{catboost.train}} (see its
#' \strong{Distributed training} section) or to
#' \code{\link{catboost.select_features}}. \code{file_with_hosts} is a plain
#' text file with one \code{host:port} line per worker, where each
#' \code{port} matches that worker's own \code{node_port} below.
#' @param node_port TCP port for this worker. Must match this worker's line
#' in the master's \code{file_with_hosts} file.
#'
#' Default value: Required argument
#' @param thread_count The number of threads used by this worker.
#'
#' Default value: number of CPU cores (\code{parallel::detectCores()}),
#' falling back to \code{1} when that cannot be determined (returns \code{NA}
#' on some platforms).
#' @return No value is returned; this call blocks until the worker stops.
#' @examples
#' \dontrun{
#' # Run on each worker machine; blocks until killed or stopped by the master.
#' catboost.run_worker(node_port = 12345)
#' }
#' @export
#' @seealso \url{https://catboost.ai/docs/features/distributed-training.html}
catboost.run_worker <- function(node_port, thread_count = parallel::detectCores()) {
    # P8 fix (catboost-8z4.103 followup): thread_count's default expression
    # (parallel::detectCores()) can itself return NA on some platforms, which
    # is not a user error -- fall back to 1 only when the caller relied on
    # the default. missing() must be checked before is.na() forces evaluation
    # of the default, so an explicitly-passed NA still falls through to the
    # validation below and errors like any other bad value.
    if (missing(thread_count) && is.na(thread_count)) {
        thread_count <- 1L
    }

    if (!is.numeric(node_port) || length(node_port) != 1L || is.na(node_port) ||
            node_port != as.integer(node_port) || node_port < 1L || node_port > 65535L) {
        stop("Parameter 'node_port' must be a single integer in 1:65535, got: ", node_port)
    }
    node_port <- as.integer(node_port)

    if (!is.numeric(thread_count) || length(thread_count) != 1L || is.na(thread_count) ||
            thread_count != as.integer(thread_count) || thread_count < 1L) {
        stop("Parameter 'thread_count' must be a single positive integer, got: ", thread_count)
    }
    thread_count <- as.integer(thread_count)

    invisible(.Call("CatBoostRunWorker_R", node_port, thread_count))
}

# P5.2 (catboost-8z4.59): param_grid accepts either a single named list
# (param name -> vector of values to try) or an unnamed list of such named
# lists (multiple grids, spans explored independently), matching Python's
# dict-or-list-of-dicts param_grid (core.py grid_search: `if
# isinstance(param_grid, Mapping): param_grid = [param_grid]`). The native
# GridSearch/RandomizedSearch entry points always expect a JSON array of
# grid objects (_catboost.pyx _PreprocessGrids.__init__: `dumps(prepared_grids)`
# where prepared_grids is a list), so a single named list is wrapped here.
prepare_grid_json <- function(param_grid) {
    if (!is.null(names(param_grid)) && all(nzchar(names(param_grid)))) {
        param_grid <- list(param_grid)
    }
    # digits = NA: jsonlite's full-round-trip-precision mode. digits = 10
    # (jsonlite's default) truncates grid values to 10 significant digits
    # before they ever reach native code (catboost-8z4.65 bug class).
    return(jsonlite::toJSON(param_grid, auto_unbox = FALSE, digits = NA))
}

#' @name catboost.grid_search
#' @title Exhaustive search over specified parameter values.
#' @description R equivalent of Python's \code{CatBoost.grid_search}: calls the
#' same native search entry point (\code{NCB::GridSearch}, vendored at
#' \code{catboost/private/libs/hyperparameter_tuning/hyperparameter_tuning.h})
#' that Python's \code{CatBoost._tune_hyperparams} calls via its Cython wrapper,
#' rather than approximating the search with an R-side loop over
#' \code{catboost.cv} -- the grid/quantization-parameter enumeration order and
#' train/test-split reuse live entirely in that native code, so calling it
#' directly is what makes best-params/best-score match the Python oracle.
#' @param param_grid Named list of parameter name -> vector of values to try, or
#' an unnamed list of such named lists (multiple grids, explored independently).
#'
#' Default value: Required argument
#' @param pool Data to search on (a \code{catboost.Pool}).
#'
#' Default value: Required argument
#' @param params Fixed parameters for \code{catboost.train}, held constant
#' across the search.
#'
#' Default value: \code{list()}
#' @param cv Number of cross-validation folds.
#'
#' Default value: 3
#' @param partition_random_seed The random seed used for splitting the data.
#'
#' Default value: 0
#' @param calc_cv_statistics Whether to estimate quality via cross-validation
#' with the found best parameters. Only used when
#' \code{search_by_train_test_split = TRUE}.
#'
#' Default value: \code{TRUE}
#' @param search_by_train_test_split If \code{TRUE}, the dataset is split into
#' train/test parts, candidates are trained on the train part and compared by
#' loss on the test part. If \code{FALSE}, every candidate is evaluated with
#' cross-validation instead.
#'
#' Default value: \code{TRUE}
#' @param refit If \code{TRUE}, fit a model on \code{pool} with the best found
#' parameters (via \code{catboost.train}) and return it as \code{$model}.
#'
#' Default value: \code{TRUE}
#' @param shuffle Shuffle the dataset objects before searching.
#'
#' Default value: \code{TRUE}
#' @param stratified Perform stratified sampling for the cross-validation
#' statistics. Unlike Python (which auto-detects this from the loss function),
#' R always defaults to \code{FALSE}, matching \code{catboost.cv}'s own
#' precedent of leaving stratification to the caller.
#'
#' Default value: \code{FALSE}
#' @param train_size Proportion of the dataset used for the train split (used
#' when \code{search_by_train_test_split = TRUE}).
#'
#' Default value: 0.8
#' @param verbose Whether to print search progress.
#'
#' Default value: \code{TRUE}
#' @param custom_objective A user-defined loss function, used instead of
#' \code{params$loss_function} for both the search itself and (when
#' \code{refit = TRUE}) the final refit. See \code{\link{catboost.train}}'s
#' \code{custom_objective} argument for the full contract.
#'
#' Default value: NULL (use \code{params$loss_function} as-is)
#' @param custom_eval_metric_object A user-defined eval metric, used instead
#' of \code{params$eval_metric} for both the search itself and (when
#' \code{refit = TRUE}) the final refit. See \code{\link{catboost.train}}'s
#' \code{custom_eval_metric_object} argument for the full contract.
#'
#' Default value: NULL (use \code{params$eval_metric} as-is)
#' @return A list with \code{$params} (best found parameters, as a named list),
#' \code{$cv_results} (a \code{data.frame} of cross-validation results with the
#' same columns \code{catboost.cv} returns), and, if \code{refit = TRUE},
#' \code{$model} (a \code{catboost.Model} fit with the best parameters).
#' @examples
#' # (c) refit = TRUE correctly reuses the supplied custom_objective for the
#' # final refit, not just for the search itself.
#' \dontrun{
#' n <- 50
#' features <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
#' label <- with(features, x1 - x2 + rnorm(n, sd = 0.3))
#' pool <- catboost.load_pool(features, label = label)
#' rmse_custom_objective <- list(
#'     calc_ders_range = function(approx, target, weight) {
#'         w <- if (is.null(weight)) 1 else weight
#'         cbind(w * (target - approx), w * (-1))
#'     })
#' result <- catboost.grid_search(
#'     list(depth = c(3, 4)), pool,
#'     params = list(iterations = 10, logging_level = "Silent",
#'                   eval_metric = "RMSE", boost_from_average = FALSE),
#'     cv = 3, refit = TRUE, verbose = FALSE,
#'     custom_objective = rmse_custom_objective)
#' # result$model was refit with rmse_custom_objective, not a built-in loss.
#' }
#' @export catboost.grid_search
catboost.grid_search <- function(param_grid,
                                  pool,
                                  params = list(),
                                  cv = 3,
                                  partition_random_seed = 0,
                                  calc_cv_statistics = TRUE,
                                  search_by_train_test_split = TRUE,
                                  refit = TRUE,
                                  shuffle = TRUE,
                                  stratified = FALSE,
                                  train_size = 0.8,
                                  verbose = TRUE,
                                  custom_objective = NULL,
                                  custom_eval_metric_object = NULL) {
    if (!inherits(pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(pool))
    if (is.null.handle(pool))
        stop("'pool' object is invalid.")

    grid_json <- prepare_grid_json(param_grid)
    fit_params <- process_synonyms(params)
    # P6.5 (catboost-8z4.94): same validation/defaulting as catboost.train,
    # applied before serialization so the search itself (not just a refit)
    # uses the custom objective/metric.
    fit_params <- apply_custom_objective_params(fit_params, custom_objective)
    fit_params <- apply_custom_eval_metric_params(fit_params, custom_eval_metric_object)
    json_params <- prepare_train_export_parameters(fit_params)

    result <- .Call("CatBoostGridSearch_R", grid_json, pool, json_params,
                     as.integer(cv), as.integer(partition_random_seed),
                     shuffle, stratified, train_size,
                     search_by_train_test_split, calc_cv_statistics, as.integer(verbose),
                     custom_objective, custom_eval_metric_object)

    best_params <- jsonlite::fromJSON(result$params)
    search_result <- list(params = best_params, cv_results = data.frame(result$cv_results))

    if (refit) {
        # P6.5 (catboost-8z4.94): forward the same custom objective/metric to
        # the refit -- otherwise the returned $model would silently train with
        # a built-in loss while the search that chose best_params used the
        # custom one (a silent correctness bug, not just a missing feature).
        search_result$model <- catboost.train(pool, params = modifyList(fit_params, best_params),
                                               custom_objective = custom_objective,
                                               custom_eval_metric_object = custom_eval_metric_object)
    }

    return(search_result)
}

#' @name catboost.randomized_search
#' @title Randomized search on hyper parameters.
#' @description R equivalent of Python's \code{CatBoost.randomized_search}:
#' calls the same native search entry point (\code{NCB::RandomizedSearch},
#' vendored at
#' \code{catboost/private/libs/hyperparameter_tuning/hyperparameter_tuning.h})
#' that Python's \code{CatBoost._tune_hyperparams} calls via its Cython wrapper.
#' In contrast to \code{catboost.grid_search}, not all parameter values are
#' tried: a fixed number (\code{n_iter}) of settings is sampled uniformly from
#' \code{param_distributions} (the native sampler seeds its shuffle from a fixed
#' constant, so the sampled combinations -- not just the winner -- are
#' reproducible across calls with the same grid and \code{n_iter}).
#'
#' Sampling from continuous distributions (Python's \code{scipy.stats}-style
#' \code{rvs()} objects) is not supported: \code{param_distributions} entries
#' must be plain value vectors, sampled uniformly by the native code.
#' @param param_distributions Named list of parameter name -> vector of values
#' to sample from, or an unnamed list of such named lists.
#'
#' Default value: Required argument
#' @param pool Data to search on (a \code{catboost.Pool}).
#'
#' Default value: Required argument
#' @param params Fixed parameters for \code{catboost.train}, held constant
#' across the search.
#'
#' Default value: \code{list()}
#' @param cv Number of cross-validation folds.
#'
#' Default value: 3
#' @param n_iter Number of parameter settings sampled.
#'
#' Default value: 10
#' @param partition_random_seed The random seed used for splitting the data.
#'
#' Default value: 0
#' @param calc_cv_statistics Whether to estimate quality via cross-validation
#' with the found best parameters. Only used when
#' \code{search_by_train_test_split = TRUE}.
#'
#' Default value: \code{TRUE}
#' @param search_by_train_test_split If \code{TRUE}, the dataset is split into
#' train/test parts, candidates are trained on the train part and compared by
#' loss on the test part. If \code{FALSE}, every candidate is evaluated with
#' cross-validation instead.
#'
#' Default value: \code{TRUE}
#' @param refit If \code{TRUE}, fit a model on \code{pool} with the best found
#' parameters (via \code{catboost.train}) and return it as \code{$model}.
#'
#' Default value: \code{TRUE}
#' @param shuffle Shuffle the dataset objects before searching.
#'
#' Default value: \code{TRUE}
#' @param stratified Perform stratified sampling for the cross-validation
#' statistics. Unlike Python (which auto-detects this from the loss function),
#' R always defaults to \code{FALSE}, matching \code{catboost.cv}'s own
#' precedent of leaving stratification to the caller.
#'
#' Default value: \code{FALSE}
#' @param train_size Proportion of the dataset used for the train split (used
#' when \code{search_by_train_test_split = TRUE}).
#'
#' Default value: 0.8
#' @param verbose Whether to print search progress.
#'
#' Default value: \code{TRUE}
#' @param custom_objective A user-defined loss function, used instead of
#' \code{params$loss_function} for both the search itself and (when
#' \code{refit = TRUE}) the final refit. See \code{\link{catboost.train}}'s
#' \code{custom_objective} argument for the full contract.
#'
#' Default value: NULL (use \code{params$loss_function} as-is)
#' @param custom_eval_metric_object A user-defined eval metric, used instead
#' of \code{params$eval_metric} for both the search itself and (when
#' \code{refit = TRUE}) the final refit. See \code{\link{catboost.train}}'s
#' \code{custom_eval_metric_object} argument for the full contract.
#'
#' Default value: NULL (use \code{params$eval_metric} as-is)
#' @return A list with \code{$params} (best found parameters, as a named list),
#' \code{$cv_results} (a \code{data.frame} of cross-validation results with the
#' same columns \code{catboost.cv} returns), and, if \code{refit = TRUE},
#' \code{$model} (a \code{catboost.Model} fit with the best parameters).
#' @export catboost.randomized_search
catboost.randomized_search <- function(param_distributions,
                                        pool,
                                        params = list(),
                                        cv = 3,
                                        n_iter = 10,
                                        partition_random_seed = 0,
                                        calc_cv_statistics = TRUE,
                                        search_by_train_test_split = TRUE,
                                        refit = TRUE,
                                        shuffle = TRUE,
                                        stratified = FALSE,
                                        train_size = 0.8,
                                        verbose = TRUE,
                                        custom_objective = NULL,
                                        custom_eval_metric_object = NULL) {
    if (!inherits(pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(pool))
    if (is.null.handle(pool))
        stop("'pool' object is invalid.")
    if (n_iter <= 0)
        stop("n_iter should be a positive number")

    grid_json <- prepare_grid_json(param_distributions)
    fit_params <- process_synonyms(params)
    # P6.5 (catboost-8z4.94): same validation/defaulting as catboost.train,
    # applied before serialization so the search itself (not just a refit)
    # uses the custom objective/metric.
    fit_params <- apply_custom_objective_params(fit_params, custom_objective)
    fit_params <- apply_custom_eval_metric_params(fit_params, custom_eval_metric_object)
    json_params <- prepare_train_export_parameters(fit_params)

    result <- .Call("CatBoostRandomizedSearch_R", grid_json, pool, json_params,
                     as.integer(n_iter), as.integer(cv), as.integer(partition_random_seed),
                     shuffle, stratified, train_size,
                     search_by_train_test_split, calc_cv_statistics, as.integer(verbose),
                     custom_objective, custom_eval_metric_object)

    best_params <- jsonlite::fromJSON(result$params)
    search_result <- list(params = best_params, cv_results = data.frame(result$cv_results))

    if (refit) {
        # P6.5 (catboost-8z4.94): forward the same custom objective/metric to
        # the refit -- see catboost.grid_search's identical fix for why.
        search_result$model <- catboost.train(pool, params = modifyList(fit_params, best_params),
                                               custom_objective = custom_objective,
                                               custom_eval_metric_object = custom_eval_metric_object)
    }

    return(search_result)
}

#' @name catboost.select_features
#' @title Select the best features by recursive elimination.
#' @description R equivalent of Python's \code{CatBoost.select_features}: trains
#' repeatedly while eliminating the weakest features, and reports which features
#' survived.
#'
#' This calls the same native entry point Python's \code{select_features} calls
#' (\code{NCB::SelectFeatures}, vendored at
#' \code{catboost/libs/features_selection/select_features.h}), so the
#' elimination loop, its per-step retraining, the SHAP-based feature strengths
#' and the final model all come from vendor code rather than an R-side
#' reimplementation.
#'
#' This is a different algorithm from \code{\link{catboost.eval_feature}}, which
#' scores caller-supplied feature sets by cross-validation instead of
#' eliminating features.
#'
#' Feature selection by feature \emph{tags} (Python's \code{grouping = "ByTags"})
#' is supported via \code{grouping}/\code{features_tags_for_select}/
#' \code{num_features_tags_to_select}: attach tags to \code{learn_pool} with
#' \code{\link{catboost.load_pool}}'s \code{feature_tags} argument, then select
#' among them here.
#' @param learn_pool The dataset to select features on (a \code{catboost.Pool}).
#'
#' Default value: Required argument
#' @param features_for_select (for \code{grouping = "Individual"}, the default)
#' Which features may be eliminated. A vector of 0-based feature indices or of
#' feature names, or a single string in the CLI's range syntax
#' (\code{"0,2-4,17"}, both ends of a range inclusive). Vectors are collapsed
#' with commas, matching Python's \code{",".join(map(str, features_for_select))}.
#' Must be \code{NULL} when \code{grouping = "ByTags"}.
#'
#' Default value: Required argument when \code{grouping = "Individual"}
#' @param num_features_to_select (for \code{grouping = "Individual"}) How many
#' features to keep out of \code{features_for_select}. Must be \code{NULL} when
#' \code{grouping = "ByTags"}.
#'
#' Default value: Required argument when \code{grouping = "Individual"}
#' @param grouping One of \code{"Individual"} (the default) or \code{"ByTags"}.
#' \code{"Individual"} selects among \code{features_for_select}; \code{"ByTags"}
#' selects among the tags attached to \code{learn_pool} via
#' \code{\link{catboost.load_pool}}'s \code{feature_tags} argument, using
#' \code{features_tags_for_select}/\code{num_features_tags_to_select} instead.
#'
#' Default value: \code{"Individual"}
#' @param features_tags_for_select (for \code{grouping = "ByTags"}) A character
#' vector of tag names (from \code{learn_pool}'s \code{feature_tags}) that may
#' be eliminated as a group. Must be \code{NULL} when \code{grouping = "Individual"}.
#'
#' Default value: Required argument when \code{grouping = "ByTags"}
#' @param num_features_tags_to_select (for \code{grouping = "ByTags"}) How many
#' tags to keep out of \code{features_tags_for_select}. Must be \code{NULL} when
#' \code{grouping = "Individual"}.
#'
#' Default value: Required argument when \code{grouping = "ByTags"}
#' @param test_pool Validation dataset used to measure the loss during
#' elimination (a \code{catboost.Pool}), or \code{NULL} to measure it on
#' \code{learn_pool}. Only one validation dataset is supported.
#'
#' Default value: \code{NULL}
#' @param params Parameters for \code{catboost.train}.
#'
#' Default value: \code{list()}
#' @param algorithm One of \code{"RecursiveByPredictionValuesChange"},
#' \code{"RecursiveByLossFunctionChange"}, \code{"RecursiveByShapValues"}.
#' \code{NULL} leaves the vendor default (\code{"RecursiveByShapValues"}).
#'
#' Default value: \code{NULL}
#' @param steps How many times a full model is trained during the elimination.
#' More steps give more accurate results. \code{NULL} leaves the vendor default
#' (1).
#'
#' Default value: \code{NULL}
#' @param shap_calc_type One of \code{"Regular"}, \code{"Approximate"},
#' \code{"Exact"}. \code{NULL} leaves the vendor default (\code{"Regular"}).
#'
#' Default value: \code{NULL}
#' @param train_final_model Whether to fit a model on the selected features and
#' return it.
#'
#' Default value: \code{TRUE}
#' @param custom_eval_metric_object A user-defined eval metric, used instead
#' of \code{params$eval_metric}. See \code{\link{catboost.train}}'s
#' \code{custom_eval_metric_object} argument for the full contract. There is
#' no \code{custom_objective} argument here: the underlying native engine
#' (\code{NCB::SelectFeatures}, \code{catboost/libs/features_selection/
#' select_features.h}) has no custom-objective-descriptor parameter at all
#' (verified against that header and \code{recursive_features_elimination.*}
#' -- zero \code{TCustomObjectiveDescriptor} references), so a
#' \code{custom_objective} argument here would be dead: it would have nothing
#' to wire it to.
#'
#' Default value: NULL (use \code{params$eval_metric} as-is)
#' @return A list with the fields of the vendor's selection summary:
#' \itemize{
#'   \item \code{selected_features} -- 0-based indices of the kept features.
#'   \item \code{selected_features_names} -- their names.
#'   \item \code{eliminated_features} -- 0-based indices of the dropped features.
#'   \item \code{eliminated_features_names} -- their names.
#'   \item \code{selected_features_tags}/\code{eliminated_features_tags} --
#'     present only when \code{grouping = "ByTags"}: the kept/dropped tag names.
#'   \item \code{loss_graph} -- list with \code{removed_features_count},
#'     \code{loss_values} and \code{main_indices} (the graph points whose loss
#'     was measured by fitting a model rather than estimated from fstr).
#'   \item \code{model} -- the fitted \code{catboost.Model}, present only when
#'     \code{train_final_model = TRUE}.
#' }
#' @export catboost.select_features
catboost.select_features <- function(learn_pool,
                                     features_for_select = NULL,
                                     num_features_to_select = NULL,
                                     test_pool = NULL,
                                     params = list(),
                                     algorithm = NULL,
                                     steps = NULL,
                                     shap_calc_type = NULL,
                                     train_final_model = TRUE,
                                     custom_eval_metric_object = NULL,
                                     grouping = NULL,
                                     features_tags_for_select = NULL,
                                     num_features_tags_to_select = NULL) {
    if (!inherits(learn_pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(learn_pool))
    if (is.null.handle(learn_pool))
        stop("'learn_pool' object is invalid.")
    if (!is.null(test_pool) && !inherits(test_pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(test_pool))
    if (!is.null(test_pool) && is.null.handle(test_pool))
        stop("'test_pool' object is invalid.")
    if (is.null(grouping))
        grouping <- "Individual"
    if (!grouping %in% c("Individual", "ByTags"))
        stop("Unsupported grouping, expecting 'Individual' or 'ByTags', got: ", grouping)
    if (length(params) == 0)
        message("Training catboost with default parameters! See help(catboost.train).")

    # Every selection knob travels inside the params JSON, exactly as Python
    # sets them on its own params dict before calling _select_features
    # (core.py:4757-4788); PlainJsonToOptions splits them back out into
    # TFeaturesSelectOptions inside the native SelectFeatures (Grouping/
    # FeaturesTagsForSelect/NumberOfFeaturesTagsToSelect: private/libs/
    # options/features_select_options.h, plain_options_helper.cpp:496-502
    # already wire "features_selection_grouping"/"features_tags_for_select"/
    # "num_features_tags_to_select" through -- no native change needed here).
    fit_params <- process_synonyms(params)
    if (grouping == "Individual") {
        if (is.null(features_for_select))
            stop("You should specify features_for_select")
        if (!is.null(features_tags_for_select))
            stop("You should not specify features_tags_for_select when grouping is Individual")
        if (is.null(num_features_to_select))
            stop("You should specify num_features_to_select")
        if (!is.null(num_features_tags_to_select))
            stop("You should not specify num_features_tags_to_select when grouping is Individual")
        fit_params$features_for_select <- paste(features_for_select, collapse = ",")
        fit_params$num_features_to_select <- as.integer(num_features_to_select)
    } else {
        if (is.null(features_tags_for_select))
            stop("You should specify features_tags_for_select")
        if (!is.null(features_for_select))
            stop("You should not specify features_for_select when grouping is ByTags")
        if (is.null(num_features_tags_to_select))
            stop("You should specify num_features_tags_to_select")
        if (!is.null(num_features_to_select))
            stop("You should not specify num_features_to_select when grouping is ByTags")
        fit_params$features_selection_grouping <- grouping
        # AsIs, same idiom prepare_train_export_parameters already uses for
        # ignored_features: avoids jsonlite's auto_unbox collapsing a
        # single-tag vector to a scalar instead of a 1-element JSON array.
        fit_params$features_tags_for_select <- I(as.character(features_tags_for_select))
        fit_params$num_features_tags_to_select <- as.integer(num_features_tags_to_select)
    }
    fit_params$train_final_model <- isTRUE(train_final_model)
    if (!is.null(algorithm))
        fit_params$features_selection_algorithm <- algorithm
    if (!is.null(steps))
        fit_params$features_selection_steps <- as.integer(steps)
    if (!is.null(shap_calc_type))
        fit_params$shap_calc_type <- shap_calc_type
    # P6.5 (catboost-8z4.94): metric only -- see custom_eval_metric_object's
    # roxygen doc above for why there is no custom_objective counterpart here.
    fit_params <- apply_custom_eval_metric_params(fit_params, custom_eval_metric_object)

    json_params <- prepare_train_export_parameters(fit_params)
    result <- .Call("CatBoostSelectFeatures_R", learn_pool, test_pool, json_params,
                    isTRUE(train_final_model), custom_eval_metric_object)

    selection <- jsonlite::fromJSON(result$summary, simplifyVector = TRUE)
    if (!is.null(result$model)) {
        raw <- .Call("CatBoostSerializeModel_R", result$model)
        model <- create.model.base(result$model, raw)
        # Deliberately not the full catboost.train post-processing: the final
        # model is fitted on a feature subset, so learn_pool's column count is
        # not its feature count and feature importances would be indexed
        # against the subset, not against learn_pool.
        model$tree_count <- catboost.ntrees(model)
        selection$model <- model
    }

    return(selection)
}

#' @name catboost.eval_feature
#' @title Evaluate the impact of feature sets.
#' @description R equivalent of the CatBoost CLI's \code{eval-feature} mode: repeated
#' cross-validated training that measures how much each tested set of features changes
#' the loss, and reports a Wilcoxon test p-value per set.
#'
#' This calls the same core entry point the CLI mode calls
#' (\code{EvaluateFeatures}, \code{catboost/libs/train_lib/eval_feature.h}), in process --
#' no CLI binary is required. There is no Python-side counterpart of this mode;
#' \code{CatBoost.select_features} is a different algorithm.
#' @param pool The dataset to evaluate on (a \code{catboost.Pool}). Test sets are not
#' supported by this mode, matching the CLI.
#'
#' Default value: Required argument
#' @param features_to_evaluate Feature sets to test, as a list of integer vectors of
#' 0-based feature indices (CLI: \code{--features-to-evaluate}). An empty list evaluates
#' the baseline only, which is meaningful with \code{eval_mode = "OneVsNone"}.
#'
#' Default value: \code{list()}
#' @param params Parameters for catboost.train.
#'
#' Default value: \code{list()}
#' @param eval_mode One of \code{"OneVsNone"}, \code{"OneVsOthers"}, \code{"OneVsAll"},
#' \code{"OthersVsAll"} (CLI: \code{--feature-eval-mode}).
#'
#' Default value: \code{"OneVsNone"}
#' @param offset First fold used for feature evaluation (CLI: \code{--offset}).
#'
#' Default value: 0
#' @param fold_count Number of folds used for feature evaluation (CLI: \code{--fold-count}).
#'
#' Default value: 3
#' @param fold_size_unit \code{"Object"} or \code{"Group"} (CLI: \code{--fold-size-unit}).
#'
#' Default value: \code{"Object"}
#' @param fold_size Fold size, in \code{fold_size_unit} units (CLI: \code{--fold-size}).
#' Exactly one of \code{fold_size} and \code{relative_fold_size} must be non-zero.
#'
#' Default value: 0
#' @param relative_fold_size Fold size as a fraction of the dataset
#' (CLI: \code{--relative-fold-size}).
#'
#' Default value: 0
#' @param timesplit_quantile Quantile for the time split (CLI: \code{--timesplit-quantile}).
#'
#' Default value: 0.5
#' @param custom_objective A user-defined loss function, used instead of
#' \code{params$loss_function}. See \code{\link{catboost.train}}'s
#' \code{custom_objective} argument for the full contract.
#'
#' Default value: NULL (use \code{params$loss_function} as-is)
#' @param custom_eval_metric_object A user-defined eval metric, used instead
#' of \code{params$eval_metric}. See \code{\link{catboost.train}}'s
#' \code{custom_eval_metric_object} argument for the full contract.
#'
#' Default value: NULL (use \code{params$eval_metric} as-is)
#' @return A list with one entry per tested feature set, mirroring the columns of the CLI's
#' \code{--feature-eval-output-file} TSV but at full double precision:
#' \itemize{
#'   \item \code{p_value} -- numeric, Wilcoxon test p-value per feature set.
#'   \item \code{best_iterations} -- list of integer vectors, best baseline iteration per fold.
#'   \item \code{metric_names} -- character vector of evaluated metric names.
#'   \item \code{metric_delta} -- numeric matrix, feature sets by metrics; average metric
#'     change, signed so that positive always means improvement.
#'   \item \code{feature_sets} -- list of integer vectors, the evaluated sets echoed back.
#' }
#' @export
catboost.eval_feature <- function(pool,
                                  features_to_evaluate = list(),
                                  params = list(),
                                  eval_mode = "OneVsNone",
                                  offset = 0,
                                  fold_count = 3,
                                  fold_size_unit = "Object",
                                  fold_size = 0,
                                  relative_fold_size = 0,
                                  timesplit_quantile = 0.5,
                                  custom_objective = NULL,
                                  custom_eval_metric_object = NULL) {

    if (!inherits(pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(pool))
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    if (!is.list(features_to_evaluate))
        stop("features_to_evaluate must be a list of integer vectors of 0-based feature indices.")
    if ((fold_size > 0) == (relative_fold_size > 0))
        stop("Exactly one of fold_size and relative_fold_size must be positive.")
    if (length(params) == 0)
        message("Training catboost with default parameters! See help(catboost.train).")

    if (offset < 0)
        stop("offset must be non-negative.")
    if (fold_count < 1)
        stop("fold_count must be positive.")

    features_to_evaluate <- lapply(features_to_evaluate, as.integer)

    # P6.5 (catboost-8z4.94): same validation/defaulting as catboost.train.
    params <- apply_custom_objective_params(params, custom_objective)
    params <- apply_custom_eval_metric_params(params, custom_eval_metric_object)

    # catboost-8z4.100: like catboost.cv, this function does not document or
    # test accepting process_synonyms() alias spellings (e.g. 'eta'), so it
    # skips resolution and validates the raw params directly -- every alias
    # name is itself a member of .catboostr_known_params, so this still
    # accepts both canonical and alias spellings without requiring resolution.
    validate_params_keys(params)

    json_params <- prepare_train_export_parameters(params)
    return(.Call("CatBoostEvaluateFeatures_R", json_params, pool,
                 features_to_evaluate, eval_mode,
                 as.integer(offset), as.integer(fold_count),
                 fold_size_unit, as.integer(fold_size),
                 as.numeric(relative_fold_size), as.numeric(timesplit_quantile),
                 custom_objective, custom_eval_metric_object))
}

#' @name catboost.model_based_eval
#' @title Model-based feature evaluation.
#' @description R equivalent of the CatBoost CLI's \code{model-based-eval} mode: continue a
#' pre-trained baseline model's training in repeated short experiments, with and without the
#' tested feature sets, and write the resulting per-experiment error logs into
#' \code{train_dir}.
#'
#' This calls the same core entry point the CLI mode calls
#' (\code{ModelBasedEval}, \code{catboost/libs/train_lib/train_model.h}), in process -- no
#' CLI binary is required. There is no Python-side counterpart of this mode.
#'
#' \strong{This mode is GPU-only.} CatBoost's CPU trainer refuses it outright
#' ("Model based eval is not implemented for CPU"), and the mode's options are only
#' recognised when \code{task_type = "GPU"}. On a build or machine without a working CUDA
#' device the call fails with a specific error rather than falling back to CPU.
#'
#' Unlike \code{\link{catboost.eval_feature}} this function takes dataset \emph{file paths}
#' rather than a \code{catboost.Pool}: the only public core entry point for this mode is
#' file-driven, it reads the baseline model's training snapshot from disk, and it writes its
#' results to disk.
#' @param learn_set Path to the learn dataset file (CLI: \code{--learn-set}).
#'
#' Default value: Required argument
#' @param test_set Path to the test dataset file (CLI: \code{--test-set}). Required: the mode
#' evaluates the change in \emph{test} error.
#'
#' Default value: Required argument
#' @param features_to_evaluate Feature sets to test, in the CLI's
#' \code{--features-to-evaluate} syntax: sets separated by \code{;}, each set a comma-separated
#' list of 0-based indices, index ranges (\code{4,78-89,312}), or feature names. \code{#tag}
#' references are not supported by this R wrapper: this function never sets
#' \code{poolLoadParams.PoolMetaInfoPath}, which tag resolution requires, so any \code{#tag}
#' value errors with "There is no tag '#x' in pool metainfo".
#'
#' Default value: Required argument
#' @param baseline_model_snapshot Path to the snapshot of the baseline model's training
#' (CLI: \code{--baseline-model-snapshot}).
#'
#' Default value: \code{"baseline_model_snapshot"}
#' @param column_description Path to the column description file (CLI:
#' \code{--column-description}). \code{NULL} means none.
#'
#' Default value: \code{NULL}
#' @param params Parameters for catboost.train. Must include \code{task_type = "GPU"};
#' \code{train_dir} selects where the results are written.
#'
#' Default value: \code{list(task_type = "GPU")}
#' @param offset Number of last iterations of the baseline model's training to evaluate over
#' (CLI: \code{--offset}). Must be at least \code{experiment_count * experiment_size}.
#'
#' Default value: 1000
#' @param experiment_count Number of experiments (CLI: \code{--experiment-count}).
#'
#' Default value: 200
#' @param experiment_size Number of iterations in one experiment (CLI:
#' \code{--experiment-size}).
#'
#' Default value: 5
#' @param use_evaluated_features_in_baseline_model Keep the evaluated features in the baseline
#' model instead of zeroing them out (CLI:
#' \code{--use-evaluated-features-in-baseline-model}).
#'
#' Default value: FALSE
#' @param delimiter Field delimiter of the dataset files (CLI: \code{--delimiter}).
#'
#' Default value: \code{"\t"}
#' @param has_header Whether the dataset files have a header line (CLI: \code{--has-header}).
#'
#' Default value: FALSE
#' @return The \code{train_dir} the per-experiment error logs were written to, invisibly.
#' @seealso \code{\link{catboost.eval_feature}} for the CPU-capable \code{eval-feature} mode.
#' @export
catboost.model_based_eval <- function(learn_set,
                                      test_set,
                                      features_to_evaluate,
                                      baseline_model_snapshot = "baseline_model_snapshot",
                                      column_description = NULL,
                                      params = list(task_type = "GPU"),
                                      offset = 1000,
                                      experiment_count = 200,
                                      experiment_size = 5,
                                      use_evaluated_features_in_baseline_model = FALSE,
                                      delimiter = "\t",
                                      has_header = FALSE) {

    if (!is.character(learn_set) || length(learn_set) != 1L)
        stop("learn_set must be a single file path.")
    if (!is.character(test_set) || length(test_set) != 1L)
        stop("test_set must be a single file path.")
    if (!file.exists(learn_set))
        stop("learn_set file does not exist: ", learn_set)
    if (!file.exists(test_set))
        stop("test_set file does not exist: ", test_set)
    if (!is.character(features_to_evaluate) || length(features_to_evaluate) != 1L ||
        !nzchar(features_to_evaluate))
        stop("features_to_evaluate must be a single non-empty string in the CLI's ",
             "--features-to-evaluate syntax, e.g. \"0,3-5;7\".")
    if (!is.null(column_description) && !file.exists(column_description))
        stop("column_description file does not exist: ", column_description)
    # TModelBasedEvalOptions::Validate(), model_based_eval_options.cpp:70. Checked here as
    # well so the failure arrives before the datasets are loaded.
    if (experiment_count * experiment_size > offset)
        stop("offset must be greater than or equal to experiment_count * experiment_size.")

    params$features_to_evaluate <- features_to_evaluate
    params$baseline_model_snapshot <- baseline_model_snapshot
    params$offset <- as.integer(offset)
    params$experiment_count <- as.integer(experiment_count)
    params$experiment_size <- as.integer(experiment_size)
    params$use_evaluated_features_in_baseline_model <-
        as.logical(use_evaluated_features_in_baseline_model)

    # catboost-8z4.100: like catboost.cv, this function does not document or
    # test accepting process_synonyms() alias spellings, so it skips
    # resolution and validates the final params directly -- every alias name
    # is itself a member of .catboostr_known_params, so this still accepts
    # both canonical and alias spellings without requiring resolution.
    validate_params_keys(params)

    json_params <- prepare_train_export_parameters(params)
    .Call("CatBoostModelBasedEval_R", json_params,
          learn_set, test_set,
          if (is.null(column_description)) "" else column_description,
          delimiter, has_header)

    return(invisible(if (is.null(params$train_dir)) "catboost_info" else params$train_dir))
}

#' @name catboost.sum_models
#' @title Sum models.
#' @description Blend trees and counters of two or more trained CatBoost models into a new model.
#'              Leaf values can be individually weighted for each input model. For example, it may
#'              be useful to blend models trained on different validation datasets.
#'
#' @param models Models for the summation.
#'
#' Default value: Required argument
#' @param weights The weights of the models.
#'
#' Default value: NULL (use weight 1 for every model)
#' @param ctr_merge_policy The counters merging policy.
#' Possible values:
#' \itemize{
#'   \item 'FailIfCtrIntersects'
#'     Ensure that the models have zero intersecting counters
#'   \item 'LeaveMostDiversifiedTable'
#'     Use the most diversified counters by the count of unique hash values
#'   \item 'IntersectingCountersAverage'
#'     Use the average ctr counter values in the intersecting bins
#' }
#'
#' Default value: 'IntersectingCountersAverage'
#' @return Model object.
#' @export
catboost.sum_models <- function(models, weights = NULL, ctr_merge_policy = 'IntersectingCountersAverage') {
    if (is.null(weights)) {
        weights = rep(1.0, length(models));
    } else if (length(models) != length(weights)) {
        stop("The length of this list must be equal to the number of blended models.");
    }

    i <- 1L
    modelsVector <- list()
    for (model in models) {
        catboost.restore_handle(model)
        modelsVector[[i]] <- model$cpp_obj$handle
        i <- i + 1L
    }
    handle <- .Call("CatBoostSumModels_R", modelsVector, weights, ctr_merge_policy)
    raw <- .Call("CatBoostSerializeModel_R", handle)
    model <- create.model.base(handle, raw)

    model$random_seed <- 0
    model$learning_rate <- 0

    return(model)
}

#' @name catboost.load_model
#' @title Load the model
#'
#' @description Load the model from a file.
#'
#'              Note: Feature importance (see \url{https://catboost.ai/docs/concepts/fstr.html#fstr})
#'              is not saved when using this function.
#' @param model_path The path to the model.
#'
#' Default value: Required argument
#' @param file_format Format of the model file.
#'
#' Default value: 'cbm'
#' @return A model object.
#' @export
#' @seealso \url{https://catboost.ai/docs/concepts/r-reference_catboost-load_model.html}
catboost.load_model <- function(model_path, file_format = "cbm") {
    model_path <- path.expand(model_path)
    handle <- .Call("CatBoostReadModel_R", model_path, file_format)
    raw <- .Call("CatBoostSerializeModel_R", handle)
    model <- create.model.base(handle, raw)
    model$tree_count <- catboost.ntrees(model)
    model$learning_rate <- catboost.get_plain_params(model)[['learning_rate']]
    return(model)
}


#' @name catboost.save_model
#' @title Save the model
#'
#' @description Save the model to a file.
#'
#'              Note: Feature importance (see \url{https://catboost.ai/docs/concepts/fstr.html#fstr})
#'              is not saved when using this function.
#' @param model The model to be saved.
#'
#' Default value: Required argument
#' @param model_path The path to the resulting binary file with the model description.
#' Used for solving other machine learning problems (for instance, applying a model).
#'
#' Default value: Required argument
#' @param file_format specified format model from a file.
#' Possible values:
#' \itemize{
#'   \item 'cbm'
#'     For catboost binary format
#'   \item 'coreml'
#'     To export into Apple CoreML format
#'   \item 'onnx'
#'     To export into ONNX-ML format
#'   \item 'pmml'
#'     To export into PMML format
#'   \item 'cpp'
#'     To export as C++ code
#'   \item 'python'
#'     To export as Python code.
#' }
#'
#' Default value: 'cbm'
#' @param export_parameters are a parameters for CoreML or PMML export.
#' @param pool is training pool.
#' @return Status, the result of model shrinking. TRUE if shrinking succeeded, FALSE otherwise.
#' @export
#' @seealso \url{https://catboost.ai/docs/features/export-model-to-core-ml.html}
catboost.save_model <- function(model, model_path,
                                file_format = "cbm",
                                export_parameters = NULL,
                                pool = NULL) {
    if (!is.null(pool) && !inherits(pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(pool))
    if (!is.null(pool) && is.null.handle(pool))
        stop("Pool object is invalid.")
    params_string <- ""
    if (!is.null(export_parameters))
        params_string <- jsonlite::toJSON(export_parameters, auto_unbox = TRUE, digits = NA)

    catboost.restore_handle(model)
    model_path <- path.expand(model_path)
    status <- .Call("CatBoostOutputModel_R", model$cpp_obj$handle, model_path, file_format, params_string, pool)
    return(status)
}


#' @name catboost.predict
#' @title Get predictions from a CatBoost model
#'
#' @description Get predictions from a CatBoost model on new data.
#'
#' In case of multiclassification the prediction is returned in the form of a matrix.
#' Each row of this matrix contains the predictions for one row of the input dataset.
#'
#' @param verbose Verbose output to stdout.
#'
#' Default value: FALSE (not used)
#' @param prediction_type The format for displaying approximated values in output data
#' (see \url{https://catboost.ai/docs/concepts/output-data.html}).
#'
#' Possible values:
#' \itemize{
#'   \item 'Probability'
#'   \item 'LogProbability'
#'   \item 'Class'
#'   \item 'RawFormulaVal'
#'   \item 'Exponent'
#'   \item 'RMSEWithUncertainty'
#' }
#'
#' Default value: 'RawFormulaVal'
#'
#' Note on R-vs-Python parity: R exposes a single \code{catboost.Model} type and a single
#' \code{catboost.predict()} entry point for every trained model, always defaulting to
#' \code{'RawFormulaVal'} regardless of the loss function used to train it. This intentionally
#' mirrors Python's base \code{CatBoost.predict()} class, whose default is likewise
#' \code{'RawFormulaVal'} (\code{core.py:2932} in the vendored Python package). Python additionally
#' offers separate scikit-learn-style estimator subclasses -- \code{CatBoostClassifier},
#' \code{CatBoostRegressor}, \code{CatBoostRanker} -- that have no equivalent object type in R and
#' that override the default independently: \code{CatBoostClassifier.predict()} defaults to
#' \code{'Class'} (\code{core.py:5552}); \code{CatBoostRegressor.predict()} resolves its default via
#' \code{_get_default_prediction_type()} (\code{core.py:6183, 6320-6329}) to \code{'Exponent'} for
#' Poisson*/Tweedie* losses, \code{'RMSEWithUncertainty'} for that loss, and \code{'RawFormulaVal'}
#' otherwise; \code{CatBoostRanker.predict()} hard-codes \code{'RawFormulaVal'} and does not accept a
#' \code{prediction_type} argument at all. Since R has no per-task subclass to hang a different
#' default off of, and since guessing the intended default from the training loss function would
#' silently change output type/shape for existing callers (e.g. returning integer class labels
#' where a numeric score was previously returned), R always defaults to \code{'RawFormulaVal'} and
#' requires classification/ranking callers to pass \code{prediction_type} explicitly (e.g.
#' \code{'Class'} or \code{'Probability'}) to obtain the same value Python's estimator subclasses
#' return by default.
#' @param ntree_start Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
#'
#' Default value: 0
#' @param ntree_end Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
#'
#' Default value: 0 (if value equals to 0 this parameter is ignored and ntree_end equal to tree_count)
#' @param thread_count The number of threads to use when applying the model. If -1, then the number of threads is set to the number of CPU cores.
#'
#' Allows you to optimize the speed of execution. This parameter doesn't affect results.
#'
#' Default value: 1
#' @return Vector of predictions (matrix for multi-class classification).
#' @seealso \url{https://catboost.ai/docs/concepts/r-reference_catboost-predict.html}


#' @rdname catboost.predict
#' @param object The model obtained as the result of training.
#'
#' Default value: Required argument
#' @param newdata The input data on which to make predictions. Should be a `catboost.Pool`
#' object.
#'
#' Default value: Required argument
#' @export
predict.catboost.Model <- function(object, newdata,
                                   verbose = FALSE, prediction_type = "RawFormulaVal",
                                   ntree_start = 0, ntree_end = 0, thread_count = -1) {
    if (!inherits(object, "catboost.Model"))
        stop("Expected catboost.Model, got: ", class(object))
    if (!inherits(newdata, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(newdata))
    if (is.null.handle(newdata))
        stop("Pool object is invalid.")

    catboost.restore_handle(object)
    prediction <- .Call("CatBoostPredictMulti_R", object$cpp_obj$handle, newdata,
                        verbose, prediction_type, ntree_start, ntree_end, thread_count)
    if (length(prediction) != nrow(newdata)) {
        prediction <- matrix(prediction, nrow = nrow(newdata), byrow = TRUE)
    }
    return(prediction)
}

#' @rdname catboost.predict
#'
#' @details The function `catboost.predict` is a synonym for `predict.catboost.Model`,
#' which is an S3 method (i.e. called like `predict(model, newdata)`).
#'
#' @param model The model obtained as the result of training.
#'
#' Default value: Required argument
#' @param pool The input data on which to make predictions. Should be a `catboost.Pool`
#' object.
#'
#' Default value: Required argument
#'
#' @export
#' @usage catboost.predict(
#'   model,
#'   pool,
#'   verbose = FALSE,
#'   prediction_type = "RawFormulaVal",
#'   ntree_start = 0,
#'   ntree_end = 0,
#'   thread_count = -1
#' )
catboost.predict <- function(model, pool, ...) {
    return(predict.catboost.Model(model, pool, ...))
}


#' @name catboost.staged_predict
#' @title Apply the model for each tree
#'
#' @description Apply the model to the given dataset and calculate the results for each i-th tree of the model
#'              taking into consideration only the trees in the range [1;i].
#'
#'              Peculiarities: In case of multiclassification the prediction is returned in the form of a matrix.
#'              Each line of this matrix contains the predictions for one object of the input dataset.
#' @param model The model obtained as the result of training.
#'
#' Default value: Required argument
#' @param pool The input dataset.
#'
#' Default value: Required argument
#' @param verbose Verbose output to stdout.
#'
#' Default value: FALSE (not used)
#' @param prediction_type The format for displaying approximated values in output data
#' (see \url{https://catboost.ai/docs/concepts/output-data.html}).
#'
#' Possible values:
#' \itemize{
#'   \item 'Probability'
#'   \item 'Class'
#'   \item 'RawFormulaVal'
#' }
#'
#' Default value: 'RawFormulaVal'
#'
#' Note on R-vs-Python parity: same rationale as \code{\link{catboost.predict}} -- R has a single
#' \code{catboost.staged_predict()} entry point for every model, always defaulting to
#' \code{'RawFormulaVal'} regardless of loss function, which mirrors Python's base
#' \code{CatBoost.staged_predict()} default. Python's \code{CatBoostClassifier.staged_predict()}
#' overrides its default to \code{'Class'} (\code{core.py:5699}); R does not replicate this because
#' it has no classifier-specific subclass to attach a different default to. Pass
#' \code{prediction_type = 'Class'} (or \code{'Probability'}) explicitly to match Python's
#' classifier default.
#' @param ntree_start Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
#'
#' Default value: 0
#' @param ntree_end Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
#'
#' Default value: 0 (if value equals to 0 this parameter is ignored and ntree_end equal to tree_count)
#' @param eval_period Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
#'
#' Default value: 1
#' @param thread_count The number of threads to use when applying the model. If -1, then the number of threads is set to the number of CPU cores.
#'
#' Allows you to optimize the speed of execution. This parameter doesn't affect results.
#'
#' Default value: 1
#' @return List object with predictions from one iteration.
#' @export
#' @seealso \url{https://catboost.ai/docs/concepts/r-reference_catboost-staged_predict.html}
catboost.staged_predict <- function(model, pool, verbose = FALSE, prediction_type = "RawFormulaVal",
                                    ntree_start = 0L, ntree_end = 0L, eval_period = 1, thread_count = -1) {
    if (!inherits(model, "catboost.Model"))
        stop("Expected catboost.Model, got: ", class(model))
    if (!inherits(pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(pool))
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    if (ntree_end == 0L)
        ntree_end <- model$tree_count

    current_tree_count <- ntree_start
    approx <- 0
    preds <- function() {
        current_tree_count <<- current_tree_count + eval_period
        if (current_tree_count - eval_period >= ntree_end)
            stop("StopIteration")
        catboost.restore_handle(model)
        current_approx <- as.array(.Call("CatBoostPredictMulti_R", model$cpp_obj$handle, pool,
                                         verbose, "RawFormulaVal",
                                         current_tree_count - eval_period,
                                         min(current_tree_count, ntree_end), thread_count))
        approx <<- approx + current_approx
        prediction_columns <- length(approx) / nrow(pool)
        loss_function <- catboost.get_model_params(model)$loss_function$type
        prediction <- .Call("CatBoostPrepareEval_R", approx, prediction_type, loss_function, prediction_columns, thread_count)
        if (prediction_columns != 1) {
            prediction <- matrix(prediction, ncol = prediction_columns, byrow = TRUE)
        }
        return(prediction)
    }

    obj <- list(nextElem = preds)
    class(obj) <- c("catboost.staged_predict", "abstractiter", "iter")
    return(obj)
}

#' @name catboost.virtual_ensembles_predict
#' @title Apply the model with several virtual ensembles
#'
#' @description Apply the model to the given dataset using several independent truncated models - virtual ensembles. Each tree in
#'              ensemble predicts its own value for each document from pool.
#'
#'              Peculiarities: Return value varies on prediction_type: array for 'VirtEnsembles' and matrix for 'TotalUncertainty'
#' @param model The model obtained as the result of training.
#'
#' Default value: Required argument
#' @param pool The input dataset.
#'
#' Default value: Required argument
#' @param verbose Verbose output to stdout.
#'
#' Default value: FALSE (not used)
#' @param prediction_type The format for displaying approximated values in output data
#' (see \url{https://catboost.ai/docs/concepts/python-reference_virtual_ensembles_predict.html#python-reference_catboostclassifier_predict__output-format}).
#'
#' Possible values:
#' \itemize{
#'   \item 'VirtEnsembles'
#'   \item 'TotalUncertainty'
#' }
#'
#' Default value: 'VirtEnsembles'
#' @param ntree_end Index of the first tree not to be used when applying the model or calculating the metrics (zero-based indexing).
#'
#' Default value: 0 (the index of the last tree to use equals to the number of trees in the model minus one)
#' @param virtual_ensembles_count Number of tree ensembles to use. Each virtual ensemble can be considered as truncated model.
#'
#' Default value: 10
#' @param thread_count The number of threads to use when applying the model. If -1, then the number of threads is set to the number of CPU cores.
#'
#' Allows you to optimize the speed of execution. This parameter doesn't affect results.
#'
#' Default value: -1
#' @return Matrix or Array of predictions (for 'TotalUncertainty' and 'VirtEnsembles' prediction_type correspondingly)
#' @export
#' @seealso \url{https://catboost.ai/docs/concepts/python-reference_virtual_ensembles_predict.html?lang=en}
catboost.virtual_ensembles_predict <- function(model, pool, verbose = FALSE, prediction_type = "VirtEnsembles",
                                    ntree_end = 0L, virtual_ensembles_count = 10, thread_count = -1) {
    catboost.restore_handle(model)
    if (!inherits(pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(pool))
    if (is.null.handle(pool))
        stop("Pool object is invalid.")

    prediction <- .Call("CatBoostPredictVirtualEnsembles_R", model$cpp_obj$handle, pool,
                        verbose, prediction_type, ntree_end, virtual_ensembles_count, thread_count)
    objects_count <- nrow(pool)
    if (prediction_type == "VirtEnsembles") {
        document_predict_size <- length(prediction) / virtual_ensembles_count / objects_count
        prediction <- aperm(
                            array(prediction,
                                    dim = c(document_predict_size, virtual_ensembles_count, objects_count)),
                            perm = c(2, 1, 3))
    } else {
        # prediction_type == "TotalUncertainty"
        prediction <- matrix(prediction, nrow = objects_count, byrow = TRUE)
    }
    return(prediction)
}


#' @name catboost.get_feature_importance
#' @title Calculate the feature importances
#'
#' @description Calculate the feature importances (see \url{https://catboost.ai/docs/concepts/fstr.html#fstr})
#'              (Regular feature importance, ShapValues, and Feature interaction strength).
#'
#' @param model The model obtained as the result of training.
#'
#' Default value: Required argument
#' @param pool The input dataset.
#'
#' The feature importance for the training dataset is calculated if this argument is not specified.
#' Models with ranking metrics require pool argument to calculate feature importance.
#'
#' Default value: NULL
#' @param type The feature importance type.
#'
#' Possible values:
#' \itemize{
#'   \item 'PredictionValuesChange'
#'
#'     Calculate score for every feature.
#'
#'   \item 'LossFunctionChange'
#'
#'     Calculate score for every feature for groupwise model.
#'
#'   \item 'FeatureImportance'
#'
#'     'LossFunctionChange' in case of groupwise model and 'PredictionValuesChange' otherwise.
#'
#'   \item 'Interaction'
#'
#'     Calculate pairwise score between every feature.
#'
#'   \item 'ShapValues'
#'
#'     Calculate SHAP Values for every object.
#'
#'   \item 'ShapInteractionValues'
#'
#'     Calculate SHAP Interaction Values between each pair of features for every object. \code{pool} is required.
#'
#'   \item 'PredictionDiff'
#'
#'     Calculate the most important features explaining the difference in predictions for a pair of documents.
#'     \code{pool} is required and must contain exactly 2 rows.
#'
#' }
#'
#' Default value: 'FeatureImportance'
#' @param thread_count The number of threads to use when applying the model. If -1, then the number of threads is set to the number of CPU cores.
#'
#' Allows you to optimize the speed of execution. This parameter doesn't affect results.
#'
#' Default value: -1
#' @param fstr_type Deprecated parameter, use 'type' instead.
#' @return Feature importances
#' @export
#' @seealso \url{https://catboost.ai/docs/features/feature-importances-calculation.html}
catboost.get_feature_importance <- function(model, pool = NULL, type = "FeatureImportance", thread_count = -1, fstr_type = NULL) {
    if (!is.null(fstr_type)) {
        type <- fstr_type
    }
    if (!inherits(model, "catboost.Model"))
        stop("Expected catboost.Model, got: ", class(model))
    if (!is.null(pool) && class(pool) != "catboost.Pool")
        stop("Expected catboost.Pool, got: ", class(pool))
    if (!is.null(pool) && is.null.handle(pool))
        stop("Pool object is invalid.")
    if ( (type == "ShapValues" || type == "LossFunctionChange" || type == "ShapInteractionValues" || type == "PredictionDiff") && length(pool) == 0)
        stop("For `", type, "` type of feature importance, the pool is required")
    if (type == "PredictionDiff" && nrow(pool) != 2)
        stop("For `PredictionDiff` type of feature importance, the pool must contain exactly 2 rows, got: ", nrow(pool))
    if ( (type == "PredictionValuesChange" || type == "FeatureImportance") && is.null(pool) && !is.null(model$feature_importances))
        return(model$feature_importances)

    catboost.restore_handle(model)
    importances <- .Call("CatBoostCalcRegularFeatureEffect_R", model$cpp_obj$handle, pool, type, thread_count)

    if (type == "Interaction") {
        colnames(importances) <- c("feature1_index", "feature2_index", "score")
    } else if (type == "ShapValues") {
        if (is.list(colnames(importances))) {
            dimnames(importances)[[length(dim(importances))]] <- c(colnames(pool), "<base>")
        }
    } else if (type == "ShapInteractionValues") {
        if (is.list(colnames(importances))) {
            nd <- length(dim(importances))
            dimnames(importances)[[nd - 1]] <- c(colnames(pool), "<base>")
            dimnames(importances)[[nd]] <- c(colnames(pool), "<base>")
        }
    } else if (type == "PredictionValuesChange" || type == "FeatureImportance" || type == "LossFunctionChange" || type == "PredictionDiff") {
        # TODO: incorrect pool and ignored_features lead to incorrect column names; testing length is not enough
        if (!is.null(pool) && dim(importances)[1] == length(colnames(pool))) {
            rownames(importances) <- colnames(pool)
        }
    } else {
        stop("Unknown type: ", type)
    }
    return(importances)
}


#' @name catboost.get_object_importance
#' @title Calculate the object importances
#'
#' @description Calculate the object importances (see \url{https://catboost.ai/docs/concepts/ostr.html}).
#'              This is the implementation of the LeafInfluence algorithm from the following paper:
#'               https://arxiv.org/pdf/1802.06640.pdf
#'
#' @param model The model obtained as the result of training.
#'
#' Default value: Required argument
#' @param pool The pool for which you want to evaluate the object importances.
#'
#' Default value: Required argument
#' @param train_pool The pool on which the model has been trained.
#'
#' Default value: Required argument
#' @param top_size Method returns the result of the top_size most important train objects. If -1, then the top size is not limited.
#'
#' Default value: -1
#' @param type
#'
#' Possible values:
#' \itemize{
#'   \item 'Average'
#'
#'     Method returns the mean train objects scores for all input objects.
#'
#'   \item 'PerObject'
#'
#'     Method returns the train objects scores for every input object.
#' }
#'
#' Default value: 'Average'
#' @param update_method Description of the update set methods are given in section 3.1.3 of the paper.
#'
#' Possible values:
#' \itemize{
#'   \item 'SinglePoint'
#'   \item 'TopKLeaves'
#'     It is posible to set top size : TopKLeaves:top=2.
#'   \item 'AllPoints'
#' }
#'
#' Default value: 'SinglePoint'
#' @param thread_count The number of threads to use when applying the model. If -1, then the number of threads is set to the number of CPU cores.
#'
#' Allows you to optimize the speed of execution. This parameter doesn't affect results.
#'
#' Default value: -1
#' @param ostr_type Deprecated parameter, use 'type' instead.
#' @return List with elements \code{"indices"} and \code{"scores"}.
#' @export
#' @seealso \url{https://catboost.ai/docs/concepts/r-reference_catboost-get_object_importance.html}
catboost.get_object_importance <- function(
    model,
    pool,
    train_pool,
    top_size = -1,
    type = "Average",
    update_method = "SinglePoint",
    thread_count = -1,
    ostr_type = NULL
) {
    if (!inherits(pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(pool))
    if (!inherits(train_pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(train_pool))
    if (is.null.handle(pool))
        stop("'pool' object is invalid.")
    if (is.null.handle(train_pool))
        stop("'train_pool' object is invalid.")
    if (top_size < 0 && top_size != -1)
        stop("top_size should be positive integer or -1.")
    catboost.restore_handle(model)
    if (!is.null(ostr_type)) {
        type <- ostr_type
        warning("ostr_type option is deprecated, use type instead")
    }
    importances <- .Call("CatBoostEvaluateObjectImportances_R", model$cpp_obj$handle,
                         pool, train_pool, top_size, type, update_method, thread_count)
    indices <- head(importances, length(importances) / 2)
    scores <- tail(importances, length(importances) / 2)
    column_count <- nrow(train_pool)
    if (top_size != -1) {
        column_count <- min(column_count, top_size)
    }
    indices <- matrix(as.integer(indices), ncol = column_count, byrow = TRUE)
    scores <- matrix(scores, ncol = column_count, byrow = TRUE)

    return(list(indices = indices, scores = scores))
}


#' @name catboost.shrink
#' @title Shrink the model
#'
#' @param model The model obtained as the result of training.
#' @param ntree_end Leave the trees with indices from the interval [ntree_start, ntree_end) (zero-based indexing).
#' @param ntree_start Leave the trees with indices from the interval [ntree_start, ntree_end) (zero-based indexing).
#'
#' @return Status, the result of model shrinking. TRUE if shrinking succeeded, FALSE otherwise.
#' @export
#' @seealso \url{https://catboost.ai/docs/concepts/r-reference_catboost-shrink.html}
catboost.shrink <- function(model, ntree_end, ntree_start = 0) {
    if (ntree_start > ntree_end)
        stop("ntree_start should be less than ntree_end.")

    catboost.restore_handle(model)
    status <- .Call("CatBoostShrinkModel_R", model$cpp_obj$handle, ntree_start, ntree_end)
    model$cpp_obj$raw <- .Call("CatBoostSerializeModel_R", model$cpp_obj$handle)
    return(status)
}


#' @name catboost.drop_unused_features
#' @title Drop unused features information from model
#'
#' @param model The model obtained as the result of training.
#'
#' @return Status, the result of dropping feature. TRUE if this succeeded, FALSE otherwise.
#' @export
catboost.drop_unused_features <- function(model) {
    catboost.restore_handle(model)
    status <- .Call("CatBoostDropUnusedFeaturesFromModel_R", model$cpp_obj$handle)
    model$cpp_obj$raw <- .Call("CatBoostSerializeModel_R", model$cpp_obj$handle)
    return(status)
}


catboost.ntrees <- function(model) {
    catboost.restore_handle(model)
    num_trees <- .Call("CatBoostGetNumTrees_R", model$cpp_obj$handle)
    return(num_trees)
}

catboost._is_groupwise_metric <- function(model) {
    catboost.restore_handle(model)
    is_groupwise_metric <- .Call("CatBoostIsGroupwiseMetric_R", model$cpp_obj$handle)
    return(is_groupwise_metric)
}


catboost._is_oblivious <- function(model) {
    catboost.restore_handle(model)
    is_oblivious <- .Call("CatBoostIsOblivious_R", model$cpp_obj$handle)
    return(is_oblivious)
}


#' @name catboost.get_model_params
#' @title Model parameters
#'
#' @description Return the model parameters.
#'
#' @param model The model obtained as the result of training.
#'
#' @return A list object with model parameters.
#' @export
#' @seealso \url{https://catboost.ai/docs/concepts/r-reference_catboost-get_model_params.html}
catboost.get_model_params <- function(model) {
    catboost.restore_handle(model)
    params <- .Call("CatBoostGetModelParams_R", model$cpp_obj$handle)
    params <- jsonlite::fromJSON(params)
    return(params)
}

#' @name catboost.get_plain_params
#' @title Plain Model parameters
#'
#' @description Return the plain model parameters.
#'
#' @param model he model obtained as the result of training.
#' @return A list object with model parameters.
#' @export
catboost.get_plain_params <- function(model) {
    catboost.restore_handle(model)
    params <- .Call("CatBoostGetPlainParams_R", model$cpp_obj$handle)
    params <- jsonlite::fromJSON(params)
    return(params)
}

#' @name catboost.get_metadata
#' @title Get model metadata
#'
#' @description Return all key/value string metadata pairs stored in the
#' model (training params, custom user data, etc). R equivalent of Python's
#' \code{model.get_metadata()} (returned as a plain named character vector
#' rather than a dict-like proxy) and of the CLI's \code{metadata dump} mode.
#' To read a single key, index the result:
#' \code{catboost.get_metadata(model)[["my_key"]]} errors on a missing key
#' (matching Python's \code{KeyError} for \code{metadata["my_key"]}), while
#' \code{catboost.get_metadata(model)["my_key"]} returns \code{NA} (matching
#' \code{metadata.get("my_key")}).
#'
#' @param model The model obtained as the result of training.
#'
#' @return A named character vector of all metadata key/value pairs.
#' @export
catboost.get_metadata <- function(model) {
    catboost.restore_handle(model)
    return(.Call("CatBoostGetModelInfo_R", model$cpp_obj$handle))
}

# P10.D (catboost-8z4.118): training-history introspection -- R equivalents
# of Python's CatBoost.best_iteration_/best_score_/evals_result_/classes_
# properties and get_best_iteration()/get_best_score()/get_evals_result()
# methods (core.py:1851-2098). CatBoost's native trainer already computes
# and records per-iteration learn/eval metric history and the early-stopping
# best iteration/score during catboost.train() (TMetricsAndTimeLeftHistory,
# libs/loggers/catboost_logger_helpers.h); TCoreModelToFullModelConverter
# unconditionally serializes it into the fitted model's own metadata under
# the "training" key (full_model_saver.cpp:572-578, ::SaveMetrics()) -- the
# same source Python's _get_best_iteration()/_get_best_score()/
# _get_metrics_evals() read from (either the in-memory TMetricsAndTimeLeftHistory
# populated during _train(), or -- after a model is loaded from disk --
# GetTrainingMetrics()'s reload of this same "training" metadata key).
# Reading this already-computed metadata is the parity target here; nothing
# below re-derives training telemetry client-side.
.catboost_get_training_metrics <- function(model) {
    training_json <- catboost.get_metadata(model)["training"]
    if (is.na(training_json) || !nzchar(training_json))
        return(NULL)
    training <- jsonlite::fromJSON(training_json, simplifyVector = FALSE)
    training$metrics
}

.catboost_eval_set_name <- function(test_index, test_count) {
    if (test_count > 1) paste0("validation_", test_index - 1) else "validation"
}

# Transposes a per-iteration list of named metric->value lists (as stored in
# LearnMetricsHistory/TestMetricsHistory) into a named list of
# metric->numeric-vector-across-iterations, matching Python's evals_result_
# shape (dict of dicts of per-iteration lists).
.catboost_transpose_metrics_history <- function(per_iteration) {
    metric_names <- unique(unlist(lapply(per_iteration, names)))
    result <- list()
    for (metric_name in metric_names) {
        values <- lapply(per_iteration, `[[`, metric_name)
        result[[metric_name]] <- as.numeric(unlist(values[!vapply(values, is.null, logical(1))]))
    }
    result
}

# R equivalent of Python's CatBoost.classes_ property (core.py:2085,
# `self._object._get_class_labels()`). Not exported as a get_ function of
# its own -- Python only exposes it as a property -- so R attaches it as a
# model-object field in create.model.base() below, matching the
# CatBoost.classes_/CatBoostClassifier.classes_/CatBoostRegressor.classes_/
# CatBoostRanker.classes_ matrix rows exactly. Delegates to native
# TFullModel::GetModelClassLabels() (model.cpp:1425) via
# CatBoostGetModelClassLabels_R, the same resolution logic (class_params/
# multiclass_params, falling back to sequential integer labels) Python's
# _get_model_class_labels() (_catboost.pyx:5342) uses -- read access to
# already-computed native state, not a second implementation of it.
.catboost_get_class_labels <- function(model) {
    catboost.restore_handle(model)
    labels_json <- .Call("CatBoostGetModelClassLabels_R", model$cpp_obj$handle)
    jsonlite::fromJSON(labels_json, simplifyVector = TRUE)
}

#' @name catboost.get_best_iteration
#' @title Get best iteration
#'
#' @description Return the 0-based iteration index of the best model seen
#' during training, as determined by early stopping / \code{use_best_model}
#' against the (last) eval set, or \code{NULL} if the model was fitted
#' without an eval set (or has no recorded training history). R equivalent
#' of Python's \code{model.get_best_iteration()}/\code{model.best_iteration_}.
#'
#' @param model The model obtained as the result of training.
#'
#' @return A single integer, or \code{NULL}.
#' @export
catboost.get_best_iteration <- function(model) {
    catboost.restore_handle(model)
    metrics <- .catboost_get_training_metrics(model)
    best_iteration <- metrics$best_iteration
    if (is.null(best_iteration)) NULL else as.integer(best_iteration)
}

#' @name catboost.get_best_score
#' @title Get best score
#'
#' @description Return the best metric value(s) seen during training for
#' the learn set and each eval set. R equivalent of Python's
#' \code{model.get_best_score()}/\code{model.best_score_}.
#'
#' @param model The model obtained as the result of training.
#'
#' @return A named list, one element per dataset (\code{learn}, plus
#' \code{validation} if there is exactly one eval set, or
#' \code{validation_0}, \code{validation_1}, ... if there are several),
#' each itself a named list of metric name to best value. Empty list if the
#' model has no recorded training history.
#' @export
catboost.get_best_score <- function(model) {
    catboost.restore_handle(model)
    metrics <- .catboost_get_training_metrics(model)
    if (is.null(metrics) || length(metrics$learn_best_error) == 0)
        return(list())
    result <- list(learn = metrics$learn_best_error)
    test_count <- length(metrics$test_best_error)
    for (test_index in seq_len(test_count)) {
        result[[.catboost_eval_set_name(test_index, test_count)]] <- metrics$test_best_error[[test_index]]
    }
    result
}

#' @name catboost.get_evals_result
#' @title Get evaluation results
#'
#' @description Return the full per-iteration metric history recorded
#' during training for the learn set and each eval set. R equivalent of
#' Python's \code{model.get_evals_result()}/\code{model.evals_result_}.
#'
#' @param model The model obtained as the result of training.
#'
#' @return A named list, one element per dataset (\code{learn}, plus
#' \code{validation} if there is exactly one eval set, or
#' \code{validation_0}, \code{validation_1}, ... if there are several),
#' each itself a named list of metric name to a numeric vector (one value
#' per iteration where the metric was recorded -- shorter than the total
#' iteration count when \code{metric_period > 1}). Empty list if the model
#' has no recorded training history.
#' @export
catboost.get_evals_result <- function(model) {
    catboost.restore_handle(model)
    metrics <- .catboost_get_training_metrics(model)
    if (is.null(metrics))
        return(list())
    result <- list()
    learn_history <- metrics$learn_metrics_history
    if (length(learn_history) > 0)
        result$learn <- .catboost_transpose_metrics_history(learn_history)
    test_history <- metrics$test_metrics_history
    if (length(test_history) > 0) {
        test_count <- max(vapply(test_history, length, integer(1)))
        for (test_index in seq_len(test_count)) {
            per_test <- lapply(test_history, function(iter_tests) {
                if (length(iter_tests) >= test_index) iter_tests[[test_index]] else list()
            })
            result[[.catboost_eval_set_name(test_index, test_count)]] <- .catboost_transpose_metrics_history(per_test)
        }
    }
    result
}

#' @name catboost.set_metadata
#' @title Set model metadata
#'
#' @description Set a single string metadata key/value pair on the model, in
#' place. R equivalent of Python's \code{model.get_metadata()[key] = value}
#' and of the CLI's \code{metadata set --key --value} mode. The change is
#' held in memory only; call \code{\link{catboost.save_model}} to persist it,
#' matching Python's calling convention.
#'
#' @param model The model obtained as the result of training.
#' @param key The metadata key name.
#' @param value The metadata value.
#'
#' @return No return value, called for side effects.
#' @export
catboost.set_metadata <- function(model, key, value) {
    catboost.restore_handle(model)
    if (!is.character(key) || length(key) != 1)
        stop("key must be a single string, got: ", class(key))
    if (!is.character(value) || length(value) != 1)
        stop("value must be a single string, got: ", class(value))
    invisible(.Call("CatBoostSetModelInfo_R", model$cpp_obj$handle, key, value))
}

#' @name catboost.get_scale_and_bias
#' @title Get model scale and bias
#'
#' @description Return the model's scale and bias, used to compute the final
#' formula as \code{Scale * sumTrees + Bias}. R equivalent of Python's
#' \code{model.get_scale_and_bias()} and of the CLI's
#' \code{normalize-model --print-scale-and-bias} mode.
#'
#' @param model The model obtained as the result of training.
#'
#' @return A list with \code{scale} (a single number) and \code{bias} (a
#' numeric vector, one value per model output dimension; empty for the
#' zero-bias default).
#' @export
catboost.get_scale_and_bias <- function(model) {
    catboost.restore_handle(model)
    return(.Call("CatBoostGetScaleAndBias_R", model$cpp_obj$handle))
}

#' @name catboost.set_scale_and_bias
#' @title Set model scale and bias
#'
#' @description Set the model's scale and bias, in place. R equivalent of
#' Python's \code{model.set_scale_and_bias(scale, bias)} and of the CLI's
#' \code{normalize-model --set-scale --set-bias} mode. The change is held in
#' memory only; call \code{\link{catboost.save_model}} to persist it, matching
#' Python's calling convention.
#'
#' @param model The model obtained as the result of training.
#' @param scale The model scale, a single number.
#' @param bias The model bias: a single number, or a numeric vector with one
#' value per model output dimension.
#'
#' @return No return value, called for side effects.
#' @export
catboost.set_scale_and_bias <- function(model, scale, bias) {
    catboost.restore_handle(model)
    if (!is.numeric(scale) || length(scale) != 1)
        stop("scale must be a single number, got: ", class(scale))
    if (!is.numeric(bias))
        stop("bias must be numeric, got: ", class(bias))
    invisible(.Call("CatBoostSetScaleAndBias_R", model$cpp_obj$handle, as.double(scale), as.double(bias)))
}

#' @name catboost.normalize_model_from_pool
#' @title Rescale a model so its raw predictions on a pool span [0, 1]
#'
#' @description CLI-only capability: R equivalent of the CLI's
#' \code{normalize-model --input-path/-i} mode (\code{mode_normalize_model.cpp}),
#' which has no Python \code{get_scale_and_bias}/\code{set_scale_and_bias}
#' counterpart. Resets the model to identity scale/bias, computes the min and
#' max of its raw (\code{RawFormulaVal}) predictions over \code{pool}, then
#' sets \code{scale = 1 / (max - min)}, \code{bias = -scale * min} so the
#' rescaled raw predictions span exactly [0, 1] -- mirroring
#' \code{mode_normalize_model.cpp}'s \code{CalcMinMaxOnAllPools} +
#' \code{model.SetScaleAndBias({scale, {bias}})} in-place.
#'
#' @param model The model obtained as the result of training.
#' @param pool A \code{catboost.Pool} (or list of pools) to compute the
#' min/max raw prediction range over.
#'
#' @return No return value, called for side effects.
#' @export
catboost.normalize_model_from_pool <- function(model, pool) {
    if (!inherits(model, "catboost.Model"))
        stop("Expected catboost.Model, got: ", class(model))
    if (inherits(pool, "catboost.Pool"))
        pool <- list(pool)
    catboost.set_scale_and_bias(model, 1.0, numeric(0))
    raw <- unlist(lapply(pool, function(p) catboost.predict(model, p, prediction_type = "RawFormulaVal")))
    mn <- min(raw)
    mx <- max(raw)
    if (mn == mx)
        stop("Model gives same result on all docs")
    scale <- 1.0 / (mx - mn)
    bias <- -scale * mn
    catboost.set_scale_and_bias(model, scale, bias)
    invisible(NULL)
}

#' @name catboost.get_model_feature_names
#' @title Get the feature names used by a model
#'
#' @description Return the names of the features used by the model (falling
#' back to their string indices for features that have no name). R
#' equivalent of Python's \code{model.feature_names_} property and of the
#' CLI's \code{metadata dump-feature-names} mode.
#'
#' @param model The model obtained as the result of training.
#'
#' @return A character vector of feature names, ordered as in the model.
#' @export
catboost.get_model_feature_names <- function(model) {
    catboost.restore_handle(model)
    return(.Call("CatBoostGetModelUsedFeatureNames_R", model$cpp_obj$handle))
}


#' @name catboost.eval_metrics
#' @title Calculate metrics.
#'
#' @description Calculate the specified metrics for the specified dataset.
#'
#' @param model The model obtained as the result of training.
#'
#' Default value: Required argument
#' @param pool The pool for which you want to evaluate the metrics.
#'
#' Default value: Required argument
#' @param metrics The list of metrics to be calculated.
#' (Supported metrics https://catboost.ai/docs/references/custom-metric__supported-metrics.html)
#'
#' Default value: Required argument
#' @param ntree_start Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
#'
#' Default value: 0
#' @param ntree_end Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
#'
#' Default value: 0 (if value equals to 0 this parameter is ignored and ntree_end equal to tree_count)
#' @param eval_period Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
#'
#' Default value: 1
#' @param thread_count The number of threads to use when applying the model.
#' If -1, then the number of threads is set to the number of CPU cores.
#'
#' Allows you to optimize the speed of execution. This parameter doesn't affect results.
#'
#' Default value: -1
#' @param tmp_dir  The name of the temporary directory for intermediate results.
#' If NULL, then the name will be generated.
#'
#' Default value: NULL
#' @return dict: metric -> array of shape [(ntree_end - ntree_start) / eval_period].
#' @export
#' @seealso \url{https://catboost.ai/docs/concepts/python-reference_catboost_eval-metrics.html}
catboost.eval_metrics <- function(model, pool, metrics, ntree_start = 0L, ntree_end = 0L,
                                  eval_period = 1, thread_count = -1, tmp_dir = NULL) {
  catboost.restore_handle(model)
  if (!inherits(pool, "catboost.Pool"))
    stop("Expected catboost.Pool, got: ", class(pool))
  if (is.null.handle(pool))
    stop("Pool object is invalid.")
  if (ntree_start < 0)
    stop("ntree_start should be greater or equal zero.")
  if (ntree_end == 0L) {
    ntree_end <- model$tree_count
  } else {
    ntree_end <- min(c(ntree_end, model$tree_count))
  }
  if (ntree_start >= ntree_end)
    stop("ntree_start should be less than ntree_end.")
  if (eval_period <= 0)
    stop("eval_period should be greater than zero.")
  if (eval_period > (ntree_end - ntree_start))
    eval_period <- ntree_end - ntree_start
  if (!is.list(metrics) && !(is.character(metrics) && length(metrics) == 1))
    stop("Unsupported metrics type, expecting list or string, got: ", typeof(metrics))
  if (is.character(metrics) && length(metrics) == 1)
    metrics <- list(metrics)
  if (length(metrics) == 0)
    stop("No metrics found.")
  if (is.null(tmp_dir))
    tmp_dir <- tempdir()
  tmp_dir <- path.expand(tmp_dir)

  params <- catboost.get_plain_params(model)
  train_dir <- params[['train_dir']]
  if (is.null(params[['train_dir']]))
    train_dir <- 'catboost_info'
  result <- .Call("CatBoostEvalMetrics_R", model$cpp_obj$handle, pool, metrics,
                  ntree_start, ntree_end, eval_period,
                  thread_count, tmp_dir, train_dir)

  return(result)
}


#' @name catboost.calc_feature_statistics
#' @title Calculate feature statistics.
#'
#' @description Get statistics for a feature using the model, dataset and target
#'              (see \url{https://catboost.ai/docs/concepts/python-reference_catboost_calc_feature_statistics.html}).
#'              The catboost model has borders for the float features used in it. The borders divide
#'              feature values into bins, and the model's prediction depends on the number of the bin
#'              where the feature value falls in.
#'
#'              For float features this function takes the model's borders and computes
#'              1) Mean target value for every bin;
#'              2) Mean model prediction for every bin;
#'              3) The number of objects in the dataset which fall into each bin;
#'              4) Predictions on varying feature: for every object, varies the feature value
#'              so that it falls into bin #0, bin #1, ... and counts model predictions, then
#'              averages that over each bin.
#'
#'              For categorical features (one-hot encoded only -- pass \code{one_hot_max_size}
#'              at training time to keep a feature one-hot) does the same, but with the
#'              feature's unique values (from \code{pool}, or \code{cat_feature_values}) taking
#'              the role of bins.
#'
#' @param model The model obtained as the result of training.
#'
#' Default value: Required argument
#' @param pool A catboost.Pool. Provides both the dataset to compute statistics on and (via
#' \code{\link{catboost.pool.get_label}}) the target.
#'
#' Default value: Required argument
#' @param feature NULL, a single feature name (character scalar) or 0-based feature index
#' (numeric scalar), or a vector/list of names/indices. NULL computes statistics for every
#' feature in \code{pool}. A single name/index returns that feature's statistics list directly;
#' anything else returns a named list of per-feature statistics lists.
#'
#' Default value: NULL
#' @param prediction_type Prediction type used for \code{mean_prediction}: one of 'Class',
#' 'Probability', 'RawFormulaVal' or 'Exponent'. If NULL, derived from the model's loss function
#' ('Probability' for CrossEntropy/Logloss, 'RawFormulaVal' otherwise).
#'
#' Default value: NULL
#' @param cat_feature_values A named list (feature name -> vector of values) of the categorical
#' feature values to compute statistics on. When \code{feature} names/selects a single
#' categorical feature, a plain (non-list) vector is also accepted for that feature. If NULL,
#' the feature's unique values already present in \code{pool} are used.
#'
#' Default value: NULL
#' @param thread_count The number of threads to use for getting statistics. If -1, then the
#' number of threads is set to the number of CPU cores.
#'
#' Default value: -1
#' @return A list (single feature) or a named list of lists (multiple features). For a float
#' feature, the list has \code{borders}, \code{binarized_feature}, \code{mean_target},
#' \code{mean_weighted_target}, \code{mean_prediction}, \code{objects_per_bin},
#' \code{predictions_on_varying_feature}. A one-hot categorical feature has the same, but
#' \code{cat_values} instead of \code{borders}.
#' @seealso \url{https://catboost.ai/docs/concepts/python-reference_catboost_calc_feature_statistics.html}
#' @export
catboost.calc_feature_statistics <- function(model, pool, feature = NULL, prediction_type = NULL,
                                              cat_feature_values = NULL, thread_count = -1) {
  if (!inherits(model, "catboost.Model"))
    stop("Expected catboost.Model, got: ", class(model))
  if (!inherits(pool, "catboost.Pool"))
    stop("Expected catboost.Pool, got: ", class(pool))
  if (is.null.handle(pool))
    stop("Pool object is invalid.")
  catboost.restore_handle(model)

  num_col <- catboost.pool.num_col(pool)
  pool_feature_names <- catboost.pool.get_feature_names(pool)

  if (is.null(prediction_type)) {
    loss_function <- catboost.get_plain_params(model)$loss_function
    prediction_type <- if (!is.null(loss_function) && loss_function %in% c("CrossEntropy", "Logloss")) {
      "Probability"
    } else {
      "RawFormulaVal"
    }
  }
  if (!(prediction_type %in% c("Class", "Probability", "RawFormulaVal", "Exponent")))
    stop('Unknown prediction type "', prediction_type, '"')

  single_feature <- !is.null(feature) && !is.list(feature) &&
    (is.character(feature) || is.numeric(feature)) && length(feature) == 1
  if (is.null(feature)) {
    features <- as.list(seq_len(num_col) - 1L)
  } else if (is.list(feature)) {
    features <- feature
  } else {
    features <- as.list(feature)
  }
  if (length(features) == 0)
    stop("feature must select at least one feature.")

  plain_cat_feature_values <- NULL
  if (!is.null(cat_feature_values) && !is.list(cat_feature_values)) {
    if (!single_feature)
      stop("cat_feature_values should be a named list when feature selects more than one feature.")
    # Deferred: keyed by the *resolved* feature name once known (below), not by
    # the raw `feature` argument -- a numeric index like `2` resolves to the
    # pool's actual feature name (e.g. "cat1"), and keying by as.character(2)
    # here would silently miss that lookup later.
    plain_cat_feature_values <- cat_feature_values
    cat_feature_values <- list()
  }
  if (is.null(cat_feature_values))
    cat_feature_values <- list()

  resolve_feature <- function(feat) {
    if (is.character(feat)) {
      idx0 <- match(feat, pool_feature_names) - 1L
      if (is.na(idx0))
        stop('No feature named "', feat, '" in model')
      name <- feat
    } else {
      idx0 <- as.integer(feat)
      if (idx0 < 0 || idx0 >= num_col)
        stop("Feature index out of range: ", feat)
      name <- if (!is.null(pool_feature_names) && length(pool_feature_names) > idx0 &&
                    pool_feature_names[idx0 + 1L] != "") {
        pool_feature_names[idx0 + 1L]
      } else {
        as.character(idx0)
      }
    }
    list(name = name, idx0 = idx0)
  }

  feature_names_out <- character(0)
  idx0_out <- integer(0)
  type_mapper <- character(0)
  cat_nums <- integer(0)
  float_nums <- integer(0)

  for (feat in features) {
    resolved <- resolve_feature(feat)
    if (resolved$name %in% feature_names_out)
      next
    type_idx <- .Call("CatBoostGetFeatureTypeAndInternalIndex_R", model$cpp_obj$handle, resolved$idx0)
    if (!(type_idx$type %in% c("float", "categorical")))
      stop("Unsupported feature type for feature '", resolved$name, "'")
    feature_names_out <- c(feature_names_out, resolved$name)
    idx0_out <- c(idx0_out, resolved$idx0)
    type_mapper <- c(type_mapper, type_idx$type)
    if (type_idx$type == "categorical") {
      cat_nums <- c(cat_nums, type_idx$index)
    } else {
      float_nums <- c(float_nums, type_idx$index)
    }
  }

  if (!is.null(plain_cat_feature_values) && length(feature_names_out) == 1) {
    cat_feature_values[[feature_names_out[1]]] <- plain_cat_feature_values
  }

  stats_list <- .Call("CatBoostGetBinarizedStatistics_R", model$cpp_obj$handle, pool,
                       as.integer(cat_nums), as.integer(float_nums), prediction_type, thread_count)

  n_cat <- length(cat_nums)
  cat_cursor <- 0L
  float_cursor <- n_cat
  statistics_by_feature <- list()

  for (i in seq_along(feature_names_out)) {
    fname <- feature_names_out[i]
    idx0 <- idx0_out[i]
    if (type_mapper[i] == "categorical") {
      cat_cursor <- cat_cursor + 1L
      stat <- stats_list[[cat_cursor]]
      internal_idx <- cat_nums[cat_cursor]
      if (!is.null(cat_feature_values[[fname]])) {
        cat_vals <- unique(as.character(cat_feature_values[[fname]]))
      } else {
        cat_vals <- .Call("CatBoostGetCatFeatureValues_R", pool, idx0)
      }
      if (length(cat_vals) > 0) {
        hashes <- vapply(
          cat_vals,
          function(v) .Call("CatBoostCalcCatFeaturePerfectHash_R", model$cpp_obj$handle, v, internal_idx),
          numeric(1)
        )
        cat_vals <- cat_vals[order(hashes)]
      }
      stat$cat_values <- cat_vals
      stat$borders <- NULL
    } else {
      float_cursor <- float_cursor + 1L
      stat <- stats_list[[float_cursor]]
    }
    statistics_by_feature[[fname]] <- stat
  }

  if (single_feature)
    return(statistics_by_feature[[feature_names_out[1]]])
  return(statistics_by_feature)
}


#' @name catboost.compare
#' @title Compare metrics of two models.
#'
#' @description Evaluate \code{metrics} for \code{model} and \code{other} on the same
#' \code{pool} and return both models' per-iteration metric values for comparison.
#'
#' Python's \code{CatBoost.compare(model, data, metrics, ...)} only draws an interactive
#' Jupyter widget from this data (it returns \code{None}); there is no headless R
#' equivalent of that widget, so \code{catboost.compare} exposes the widget's underlying
#' metrics-diff data structure instead -- what \code{compare()} computes internally via
#' \code{self._eval_metrics(...)} and \code{model._eval_metrics(...)} on the same
#' pool/metrics (both funnel into the same \code{TMetricsPlotCalcer} vendor entry point
#' as \code{\link{catboost.eval_metrics}}, which this function calls once per model).
#'
#' Named \code{catboost.compare} (not \code{model.compare}, and not an S3 method) because
#' a dotted name would collide with R's existing S3 dispatch on \code{catboost.Model}
#' (see \code{\link{predict.catboost.Model}}).
#'
#' @param model The first model obtained as a result of training.
#'
#' Default value: Required argument
#' @param other The second (other) model to compare against \code{model}.
#'
#' Default value: Required argument
#' @param pool A catboost.Pool to evaluate both models' metrics on.
#'
#' Default value: Required argument
#' @param metrics A list of metric names to be calculated.
#' (Full list of supported metrics: https://catboost.ai/docs/references/custom-metric__supported-metrics.html)
#'
#' Default value: Required argument
#' @param ntree_start Each model is applied on the interval [ntree_start, ntree_end) with the
#' step eval_period (zero-based indexing).
#'
#' Default value: 0
#' @param ntree_end Each model is applied on the interval [ntree_start, ntree_end) with the
#' step eval_period (zero-based indexing). If value equals 0, this parameter is ignored and
#' ntree_end is set to that model's own tree_count.
#'
#' Default value: 0
#' @param eval_period Each model is applied on the interval [ntree_start, ntree_end) with the
#' step eval_period (zero-based indexing).
#'
#' Default value: 1
#' @param thread_count The number of threads to use when applying each model. If -1, then the
#' number of threads is set to the number of CPU cores.
#'
#' Default value: -1
#' @param tmp_dir The name of the temporary directory for intermediate results. If NULL, the
#' name is generated with \code{tempdir()}.
#'
#' Default value: NULL
#' @return An object of class \code{catboost.compare} (a list): \code{model} and \code{other},
#' each the same named-list-of-numeric-vectors (metric name -> per-iteration values) that
#' \code{\link{catboost.eval_metrics}} returns for that model on \code{pool}.
#' @export
#' @seealso \url{https://catboost.ai/docs/concepts/python-reference_catboost_compare.html}
catboost.compare <- function(model, other, pool, metrics, ntree_start = 0L, ntree_end = 0L,
                              eval_period = 1, thread_count = -1, tmp_dir = NULL) {
  if (is.null(model))
    stop("You should provide a model for comparison.")
  if (is.null(other))
    stop("You should provide another model for comparison.")
  if (is.null(pool))
    stop("You should provide data for comparison.")
  if (is.null(metrics))
    stop("You should provide metrics for comparison.")
  if (!inherits(model, "catboost.Model"))
    stop("Expected catboost.Model, got: ", class(model))
  if (!inherits(other, "catboost.Model"))
    stop("Expected catboost.Model, got: ", class(other))

  result <- list(
    model = catboost.eval_metrics(model, pool, metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir),
    other = catboost.eval_metrics(other, pool, metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir)
  )
  class(result) <- "catboost.compare"
  return(result)
}


#' @name catboost.get_roc_curve
#' @title Build a ROC curve
#' @description Build the points of the ROC curve for a binary classification model, matching
#' Python's \code{catboost.utils.get_roc_curve} and the CLI's \code{roc} mode. Ports the same
#' engine both wrap (\code{catboost/private/libs/algo/roc_curve.cpp TRocCurve}): raw model
#' predictions are converted to probabilities, sorted in descending order, and swept once to
#' accumulate false positive / false negative rates, inserting a synthetic point wherever the
#' FPR and FNR curves cross.
#' @param model The model obtained as the result of training on a binary classification task.
#'
#' Default value: Required argument
#' @param pool A \code{catboost.Pool} (or list of \code{catboost.Pool}s) with label data, used
#' to build the curve. Labels are binarized: values >= 0.5 count as the positive class. Labels
#' that do not round to 0 or 1 (i.e. the pool is not a binary classification target) make the
#' call fail with an error.
#'
#' Default value: Required argument
#' @return A list with three numeric vectors of equal length, sorted by decreasing
#' \code{threshold}: \code{fpr} (false positive rate), \code{tpr} (true positive rate), and
#' \code{threshold} (probability decision boundary, in \code{[0, 1]}).
#' @export
#' @seealso \url{https://catboost.ai/docs/concepts/python-reference_utils_get_roc_curve.html}
catboost.get_roc_curve <- function(model, pool) {
  if (!inherits(model, "catboost.Model"))
    stop("Expected catboost.Model, got: ", class(model))
  catboost.restore_handle(model)

  pools <- if (inherits(pool, "catboost.Pool")) list(pool) else pool
  if (!is.list(pools) || length(pools) == 0)
    stop("Expected catboost.Pool or non-empty list of catboost.Pool, got: ", class(pool))

  probability <- numeric(0)
  target <- integer(0)
  for (p in pools) {
    if (!inherits(p, "catboost.Pool"))
      stop("Expected catboost.Pool, got: ", class(p))
    if (is.null.handle(p))
      stop("Pool object is invalid.")
    label <- catboost.pool.get_label(p)
    if (length(label) == 0)
      stop("Pool has no label data.")
    probability <- c(probability, catboost.predict(model, p, prediction_type = "Probability"))
    target <- c(target, as.integer(label + 0.5)) # custom round for accuracy, matches TRocCurve::BuildCurve
  }
  bad <- unique(target[!(target %in% c(0L, 1L))])
  if (length(bad) > 0)
    stop("catboost.get_roc_curve requires labels that round to 0 or 1 (binary classification); ",
         "found rounded label(s): ", paste(bad, collapse = ", "))

  count1 <- sum(target == 1L)
  count0 <- sum(target == 0L)
  if (count0 == 0 || count1 == 0)
    stop("Need documents of both classes 0 and 1 to build a ROC curve.")

  ord <- order(-probability) # stable sort, descending by probability
  probability <- probability[ord]
  target <- target[ord]

  n <- length(probability)
  fnr <- numeric(0)
  fpr <- numeric(0)
  boundary <- numeric(0)
  eps <- 1e-13
  # ponytail: O(n^2) vector growth via <<- append per boundary point; pre-allocate numeric(n) for boundary/fnr/fpr and truncate if profiling shows this is load-bearing at scale
  add_point <- function(newBoundary, newFnr, newFpr) {
    len <- length(fnr)
    if (len > 0) {
      oldFnr <- fnr[len]
      oldFpr <- fpr[len]
      if (oldFpr < oldFnr && newFpr > newFnr) {
        # will happen at least once: first point (1, 1, 0) satisfies first inequality,
        # last point (0, 0, 1) satisfies second inequality
        x1 <- boundary[len]; x2 <- newBoundary
        y11 <- oldFnr; y21 <- newFnr
        y12 <- oldFpr; y22 <- newFpr
        x <- x1 + (x1 - x2) * (y11 - y12) / ((y21 - y22) - (y11 - y12))
        if ((y22 - y12) < eps) {
          y <- 0.5 * (y12 + y22)
        } else if ((y11 - y21) < eps) {
          y <- 0.5 * (y11 + y21)
        } else {
          y <- y11 + (x1 - x) * (y21 - y11) / (x1 - x2)
        }
        boundary[len + 1] <<- x; fnr[len + 1] <<- y; fpr[len + 1] <<- y
        len <- len + 1
      }
    }
    boundary[len + 1] <<- newBoundary; fnr[len + 1] <<- newFnr; fpr[len + 1] <<- newFpr
  }

  add_point(1, 1, 0) # always starts with (1, 1, 0)
  countTarget1 <- 0L
  countTarget0 <- 0L
  for (i in seq_len(n - 1)) {
    if (target[i] == 1L) countTarget1 <- countTarget1 + 1L else countTarget0 <- countTarget0 + 1L
    if (probability[i + 1] < (probability[i] - eps)) {
      newBoundary <- 0.5 * (probability[i] + probability[i + 1])
      newFnr <- (count1 - countTarget1) / count1
      newFpr <- countTarget0 / count0
      add_point(newBoundary, newFnr, newFpr)
    }
  }
  add_point(0, 0, 1) # always ends with (0, 0, 1)

  return(list(fpr = fpr, tpr = 1 - fnr, threshold = boundary))
}


# Not exported: formats the "<hash:...>" fallback label catboost.plot_tree()'s
# resolve_cat_value() uses when a categorical split's hash can't be resolved back
# to a string. target_hash is a ui32 (OneHotFeature.Value, up to ~4.29e9) surfaced
# as an R double via as.numeric() -- doubles are exact up to 2^53, so %.0f (not
# %d/as.integer(), whose ceiling is 2147483647) is required to avoid silently
# collapsing every hash above 2^31 to the same "<hash:NA>" label.
catboost.plot_tree.format_hash_fallback <- function(target_hash) {
  sprintf("<hash:%.0f>", target_hash)
}


#' @name catboost.plot_tree
#' @title Plot a single tree's structure.
#'
#' @description Return the node/edge structure of one tree in \code{model}
#' (see \url{https://catboost.ai/docs/concepts/python-reference_catboost_plot_tree.html}).
#'
#' Python's \code{plot_tree} returns a \code{graphviz.Digraph} built from the model's
#' internal per-tree splits and leaf values (\code{_get_tree_splits}/\code{_get_tree_leaf_values}
#' in \code{catboost/python-package/catboost/core.py}, reading the same vendor
#' \code{TFullModel} tree layout that \code{\link{catboost.save_model}}'s \code{"json"} export
#' format serializes). \code{catboost.plot_tree} reuses that existing JSON export (no new
#' native entry point) to read the same split/leaf data and reconstructs the DOT-text graph
#' with base R string building -- no \code{DiagrammeR}/graphviz R package dependency is added,
#' since none is required for either the structural differential test (spec Sec 4.3's
#' "Structural" row targets canonical node/edge JSON, not a rendered image) or the "object of
#' the expected class" smoke test.
#'
#' Only oblivious (symmetric) trees with \code{FloatFeature}/\code{OneHotFeature} splits are
#' supported; other tree/split kinds stop with an explicit error rather than silently
#' mis-rendering.
#'
#' @param model The model obtained as the result of training.
#'
#' Default value: Required argument
#' @param tree_idx 0-based index of the tree to plot.
#'
#' Default value: Required argument
#' @param pool A catboost.Pool used to resolve feature names and to decode categorical split
#' values. Required if the tree splits on any categorical feature (mirrors Python's own
#' \code{plot_tree}, which raises if a categorical split is present and no pool is given);
#' optional for float-only trees, in which case node labels fall back to the 0-based flat
#' feature index -- again mirroring Python's own \code{pool = NULL} fallback. Categorical
#' split values are decoded by re-hashing the pool's raw string values with
#' \code{CatBoostCalcCatFeatureHash_R} (the same raw \code{CalcCatFeatureHash()} stored in the
#' exported JSON model's \code{TOneHotSplit::Value}) until the split's hash is matched; when
#' \code{pool} was built by \code{\link{catboost.load_pool}}/\code{\link{catboost.from_matrix}}
#' (which pre-hash categorical columns into floats before the vendor pool is built, so no
#' hash-to-string dictionary survives -- see \code{\link{catboost.calc_feature_statistics}}'s
#' docs) the original string cannot be recovered and the label falls back to
#' \code{"<hash:...>"}.
#'
#' Default value: NULL
#' @return An object of class \code{catboost.plot_tree}: a list with \code{dot} (character
#' scalar, DOT-language source text), \code{nodes} (data.frame: \code{id}, \code{label},
#' \code{color}, \code{shape}) and \code{edges} (data.frame: \code{from}, \code{to},
#' \code{label}).
#' @seealso \url{https://catboost.ai/docs/concepts/python-reference_catboost_plot_tree.html}
#' @export
catboost.plot_tree <- function(model, tree_idx, pool = NULL) {
  if (!inherits(model, "catboost.Model"))
    stop("Expected catboost.Model, got: ", class(model))
  if (!is.null(pool) && !inherits(pool, "catboost.Pool"))
    stop("Expected catboost.Pool, got: ", class(pool))
  if (!is.null(pool) && is.null.handle(pool))
    stop("Pool object is invalid.")
  if (!catboost._is_oblivious(model))
    stop("catboost.plot_tree only supports oblivious (symmetric) trees.")

  num_trees <- catboost.ntrees(model)
  tree_idx <- as.integer(tree_idx)
  if (length(tree_idx) != 1 || is.na(tree_idx) || tree_idx < 0 || tree_idx >= num_trees)
    stop("tree_idx out of range [0, ", num_trees - 1, "]: ", tree_idx)

  json_path <- tempfile(fileext = ".json")
  on.exit(unlink(json_path), add = TRUE)
  catboost.save_model(model, json_path, file_format = "json", pool = pool)
  model_json <- jsonlite::fromJSON(json_path, simplifyVector = FALSE)

  if (is.null(model_json$oblivious_trees))
    stop("catboost.plot_tree only supports oblivious (symmetric) trees.")
  tree <- model_json$oblivious_trees[[tree_idx + 1L]]
  splits <- tree$splits
  leaf_values <- tree$leaf_values

  num_leaves <- 2L^length(splits)
  if (length(leaf_values) != num_leaves)
    stop("catboost.plot_tree does not support multi-dimensional leaf values ",
         "(e.g. multiclass models).")

  float_features <- model_json$features_info$float_features
  cat_features <- model_json$features_info$categorical_features

  # NB: the JSON export's "cat_features_hash" table is empty for models trained
  # from an R-built Pool: catboost.from_matrix()/catboost.load_pool() pre-hash
  # categorical columns into floats via CatBoostHashStrings_R before the value
  # ever reaches the vendor pool builder, so the pool's ObjectsData never gets a
  # hash-to-string dictionary to export (same root cause catboost.calc_feature_statistics
  # documents for CatBoostGetCatFeatureValues_R -- see test_calc_feature_statistics.R).
  # Resolve categorical values by asking the pool for its (possibly empty) set of raw
  # string values and re-hashing each one with the raw CalcCatFeatureHash() function
  # (CatBoostCalcCatFeatureHash_R) until the split's hash (TOneHotSplit::Value, the raw
  # hash -- NOT the perfect-hash index CatBoostCalcCatFeaturePerfectHash_R returns) is
  # matched. When the pool cannot supply any candidate strings, fall back to a deterministic
  # "<hash:...>" label instead of silently mis-labelling or hard-failing the whole
  # plot for a categorical model built from an R Pool.
  resolve_cat_value <- function(flat_idx, target_hash) {
    # TOneHotSplit::Value (vendor online_ctr.h) is a signed `int` that stores the ui32
    # CalcCatFeatureHash() result bit-reinterpreted, so the JSON export's "value" field
    # comes back negative whenever the true hash is >= 2^31. CatBoostCalcCatFeatureHash_R
    # always returns the unsigned ui32, so canonicalize target_hash to the same unsigned
    # range before comparing; the (possibly negative) original is kept for the fallback
    # label so its format stays unchanged.
    target_hash_unsigned <- if (target_hash < 0) target_hash + 2^32 else target_hash
    candidates <- .Call("CatBoostGetCatFeatureValues_R", pool, flat_idx)
    for (v in candidates) {
      h <- as.numeric(.Call("CatBoostCalcCatFeatureHash_R", v))
      if (isTRUE(all.equal(h, target_hash_unsigned)))
        return(v)
    }
    catboost.plot_tree.format_hash_fallback(target_hash)
  }

  find_by_index <- function(entries, field, idx) {
    for (e in entries) {
      if (!is.null(e[[field]]) && as.integer(e[[field]]) == as.integer(idx))
        return(e)
    }
    NULL
  }

  split_label <- function(split) {
    if (identical(split$split_type, "FloatFeature")) {
      entry <- find_by_index(float_features, "feature_index", split$float_feature_index)
      if (is.null(entry))
        stop("No float feature metadata for feature_index ", split$float_feature_index)
      fid <- entry$feature_id
      name <- if (!is.null(pool) && !is.null(fid) && nzchar(fid)) fid else as.character(entry$flat_feature_index)
      paste0(name, ", value>", sprintf("%.6g", as.numeric(split$border)))
    } else if (identical(split$split_type, "OneHotFeature")) {
      if (is.null(pool))
        stop("Please pass training dataset to catboost.plot_tree function, ",
             "training dataset is required if categorical features are present in the model.")
      entry <- find_by_index(cat_features, "feature_index", split$cat_feature_index)
      if (is.null(entry))
        stop("No categorical feature metadata for feature_index ", split$cat_feature_index)
      fid <- entry$feature_id
      name <- if (!is.null(fid) && nzchar(fid)) fid else as.character(entry$flat_feature_index)
      cat_value <- resolve_cat_value(entry$flat_feature_index, as.numeric(split$value))
      paste0(name, ", value=", cat_value)
    } else {
      stop("catboost.plot_tree does not support split_type '", split$split_type, "'.")
    }
  }

  node_id <- character(0); node_label <- character(0)
  node_color <- character(0); node_shape <- character(0)
  edge_from <- character(0); edge_to <- character(0); edge_label <- character(0)

  layer_size <- 1L
  current_size <- 0L
  for (split_num in seq.int(length(splits) - 1L, -1L, by = -1L)) {
    for (node_num in seq_len(layer_size)) {
      if (split_num >= 0L) {
        label <- split_label(splits[[split_num + 1L]])
        color <- "black"; shape <- "ellipse"
      } else {
        label <- sprintf("val = %.3f\n", as.numeric(leaf_values[[node_num]]))
        color <- "red"; shape <- "rect"
      }
      node_id <- c(node_id, as.character(current_size))
      node_label <- c(node_label, label)
      node_color <- c(node_color, color)
      node_shape <- c(node_shape, shape)
      if (current_size > 0L) {
        parent <- (current_size - 1L) %/% 2L
        edge_from <- c(edge_from, as.character(parent))
        edge_to <- c(edge_to, as.character(current_size))
        edge_label <- c(edge_label, if (current_size %% 2L == 0L) "Yes" else "No")
      }
      current_size <- current_size + 1L
    }
    layer_size <- layer_size * 2L
  }

  nodes_df <- data.frame(id = node_id, label = node_label, color = node_color,
                          shape = node_shape, stringsAsFactors = FALSE)
  edges_df <- data.frame(from = edge_from, to = edge_to, label = edge_label,
                          stringsAsFactors = FALSE)

  dot_lines <- c(
    "digraph {",
    sprintf('\t%s [label="%s" color=%s shape=%s]', nodes_df$id, nodes_df$label,
            nodes_df$color, nodes_df$shape),
    if (nrow(edges_df) > 0)
      sprintf('\t%s -> %s [label=%s]', edges_df$from, edges_df$to, edges_df$label),
    "}"
  )
  dot_text <- paste(dot_lines, collapse = "\n")

  result <- list(dot = dot_text, nodes = nodes_df, edges = edges_df)
  class(result) <- "catboost.plot_tree"
  return(result)
}


#' @name catboost.restore_handle
#' @title Restore or complete model handle after de-serializing
#'
#' @description After de-serializing a model object through R base's functions (`readRDS`, `load`),
#'              its underlying object will not exist in the computer's memory anymore, and needs
#'              to be restored from the raw bytes that the model stores.
#'
#'              This is automatically done internally when calling functions such as \link{catboost.predict},
#'              but the process is repeated at each call, which makes them slower than if using a
#'              fresh model object and increases memory usage inbetween calls to the garbage collector.
#'              This function allows restoring the internal object beforehand so as to avoid
#'              restoring the object multiple times.
#'
#'              Note that the model object needs to be re-assigned as the output of this function,
#'              as the modifications are not done in-place.
#'
#' @param model The model obtained as the result of training which has been serialized and is
#'              now de-serialized.
#'
#' @return The model object with its handle pointing to a valid object in memory.
#' @export
catboost.restore_handle <- function(model) {
    if (!inherits(model, "catboost.Model"))
        stop("Expected catboost.Model, got: ", class(model))
    if (is.null.handle(model$cpp_obj$handle))
        model$cpp_obj$handle <- .Call("CatBoostDeserializeModel_R", model$cpp_obj$raw)
    return(model)
}

is.null.handle <- function(handle) {
  stopifnot(typeof(handle) == "externalptr")
  .Call("CatBoostIsNullHandle_R", handle)
}

create.model.base <- function(handle, raw) {
    model <- list(cpp_obj = as.environment(list(handle = handle, raw = raw)))
    class(model) <- "catboost.Model"
    # P10.D (catboost-8z4.118): attach training-history introspection fields
    # to every constructed model object (trained, loaded, sum_models'd, or
    # select_features' final refit), matching Python's CatBoost.classes_/
    # best_iteration_/best_score_/evals_result_ being properties available
    # on any CatBoost instance, not just a freshly-fit one.
    model$classes_ <- .catboost_get_class_labels(model)
    model$best_iteration_ <- catboost.get_best_iteration(model)
    model$best_score_ <- catboost.get_best_score(model)
    model$evals_result_ <- catboost.get_evals_result(model)
    return(model)
}
