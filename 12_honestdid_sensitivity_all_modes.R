# 12_honestdid_sensitivity_nonbridge_modes.R
#
# Formal HonestDiD sensitivity analysis for non-bridge event-study models.
#
# This script uses the full variance-covariance matrix of the event-study
# treatment-interaction coefficients. It deliberately DOES NOT construct a
# diagonal approximation from standard errors.
#
# Prerequisite:
#   Run the corresponding 10a-10d event-study scripts with covariance export
#   enabled. Each must save:
#     event_time_i, event_time_j, covariance
#
# Scope:
# subway, bus, taxi, for-hire, and bridge-and-tunnel outcomes.
# Bridge-and-tunnel results are reported separately because the event study
# uses aggregate treated and comparison facility groups with HC1 covariance.
#
# Default estimand:
#   February 2025 (event_time = 1), the first fully treated calendar month.
#   Change TARGET_EVENT_TIME to 0 only if you explicitly want the January 2025
#   implementation-month effect (which includes January 1-4 pre-policy hours).
#
# Run from the project root:
#   source("data/analysis/12_honestdid_sensitivity_nonbridge_modes.R")

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(HonestDiD)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TARGET_EVENT_TIME <- 1L
MBAR_VALUES <- seq(0.25, 2.00, by = 0.25)

dir.create("outputs/models", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)

OUT_TXT <- "outputs/models/12_honestdid_all_modes_results.txt"
OUT_COMBINED <- "outputs/models/12_honestdid_all_modes_summary.csv"

MODE_CONFIG <- list(
  subway = list(
    coefficients = "outputs/models/10a_event_study_subway_coefficients.csv",
    covariance = "outputs/models/10a_event_study_subway_covariance.csv",
    label = "Subway",
    output_summary = "outputs/models/12_honestdid_subway_summary.csv",
    output_figure = "outputs/figures/12_honestdid_subway_sensitivity.png"
  ),
  bus = list(
    coefficients = "outputs/models/10b_event_study_bus_coefficients.csv",
    covariance = "outputs/models/10b_event_study_bus_covariance.csv",
    label = "Bus",
    output_summary = "outputs/models/12_honestdid_bus_summary.csv",
    output_figure = "outputs/figures/12_honestdid_bus_sensitivity.png"
  ),
  taxi = list(
    coefficients = "outputs/models/10c_event_study_taxi_coefficients.csv",
    covariance = "outputs/models/10c_event_study_taxi_covariance.csv",
    label = "Taxi",
    output_summary = "outputs/models/12_honestdid_taxi_summary.csv",
    output_figure = "outputs/figures/12_honestdid_taxi_sensitivity.png"
  ),
  forhire = list(
    coefficients = "outputs/models/10d_event_study_forhire_coefficients.csv",
    covariance = "outputs/models/10d_event_study_forhire_covariance.csv",
    label = "For-Hire",
    output_summary = "outputs/models/12_honestdid_forhire_summary.csv",
    output_figure = "outputs/figures/12_honestdid_forhire_sensitivity.png"
  ),
  bridge = list(
    coefficients = "outputs/models/10e_event_study_bridge_coefficients.csv",
    covariance = "outputs/models/10e_event_study_bridge_covariance.csv",
    label = "Bridge and Tunnel",
    output_summary = "outputs/models/12_honestdid_bridge_summary.csv",
    output_figure = "outputs/figures/12_honestdid_bridge_sensitivity.png"
  )
)

# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

load_event_study <- function(path) {
  if (!file.exists(path)) {
    stop(paste("Missing event-study coefficient file:", path))
  }

  df <- read_csv(path, show_col_types = FALSE)

  required <- c("event_time", "beta")
  missing <- setdiff(required, names(df))

  if (length(missing) > 0) {
    stop(
      paste0(
        "Missing required columns in ", path, ": ",
        paste(missing, collapse = ", ")
      )
    )
  }

  df <- df %>%
    transmute(
      event_time = as.integer(event_time),
      beta = as.numeric(beta)
    ) %>%
    filter(!is.na(event_time), !is.na(beta)) %>%
    arrange(event_time)

  if (anyDuplicated(df$event_time) > 0) {
    stop(paste("Duplicate event_time values in:", path))
  }

  if (-1L %in% df$event_time) {
    stop(
      paste(
        "The omitted reference period (-1) must not appear among exported",
        "event-study interaction coefficients:", path
      )
    )
  }

  pre_times <- df$event_time[df$event_time < 0]
  post_times <- df$event_time[df$event_time >= 0]

  if (length(pre_times) == 0 || length(post_times) == 0) {
    stop(paste("Need both pre- and post-policy coefficients in:", path))
  }

  if (!all(diff(pre_times) == 1L)) {
    stop(paste("Pre-policy event times are not consecutive in:", path))
  }

  if (!all(diff(post_times) == 1L)) {
    stop(paste("Post-policy event times are not consecutive in:", path))
  }

  df
}


load_full_covariance <- function(path, event_times) {
  if (!file.exists(path)) {
    stop(
      paste0(
        "Missing covariance file: ", path, "\n",
        "Run the patched 10a-10d event-study script first. A formal HonestDiD ",
        "analysis cannot be run from standard errors alone."
      )
    )
  }

  cov_df <- read_csv(path, show_col_types = FALSE)

  required <- c("event_time_i", "event_time_j", "covariance")
  missing <- setdiff(required, names(cov_df))

  if (length(missing) > 0) {
    stop(
      paste0(
        "Covariance file must contain event_time_i, event_time_j, covariance. ",
        "Missing in ", path, ": ", paste(missing, collapse = ", ")
      )
    )
  }

  cov_df <- cov_df %>%
    transmute(
      event_time_i = as.integer(event_time_i),
      event_time_j = as.integer(event_time_j),
      covariance = as.numeric(covariance)
    ) %>%
    filter(
      event_time_i %in% event_times,
      event_time_j %in% event_times
    )

  expected_pairs <- expand_grid(
    event_time_i = event_times,
    event_time_j = event_times
  )

  if (nrow(cov_df) != nrow(expected_pairs) ||
      anyDuplicated(cov_df[c("event_time_i", "event_time_j")]) > 0) {
    stop(
      paste0(
        "Covariance file does not contain exactly one entry for every ",
        "event-time pair: ", path
      )
    )
  }

  missing_pairs <- anti_join(
    expected_pairs,
    cov_df,
    by = c("event_time_i", "event_time_j")
  )

  if (nrow(missing_pairs) > 0 || any(!is.finite(cov_df$covariance))) {
    stop(paste("Covariance file is incomplete or non-finite:", path))
  }

  cov_wide <- cov_df %>%
    mutate(event_time_j = as.character(event_time_j)) %>%
    pivot_wider(
      names_from = event_time_j,
      values_from = covariance
    ) %>%
    arrange(event_time_i)

  sigma <- as.matrix(
    cov_wide[, as.character(event_times), drop = FALSE]
  )
  rownames(sigma) <- as.character(cov_wide$event_time_i)
  colnames(sigma) <- as.character(event_times)

  if (!isTRUE(all.equal(
    sigma,
    t(sigma),
    tolerance = 1e-10,
    check.attributes = FALSE
  ))) {
    stop(paste("Covariance matrix is not symmetric:", path))
  }

  if (any(diag(sigma) < -1e-12)) {
    stop(paste("Covariance matrix has negative diagonal entries:", path))
  }

  sigma
}


extract_ci <- function(result_object, object_name) {
  result_df <- as.data.frame(result_object)

  lower_col <- intersect(
    c("lb", "lower", "Lower", "CI_Lower"),
    names(result_df)
  )
  upper_col <- intersect(
    c("ub", "upper", "Upper", "CI_Upper"),
    names(result_df)
  )

  if (length(lower_col) == 0 || length(upper_col) == 0) {
    stop(
      paste0(
        "Could not identify lower/upper CI columns in ", object_name,
        ". Inspect the HonestDiD package output."
      )
    )
  }

  data.frame(
    ci_low = as.numeric(result_df[[lower_col[1]]][1]),
    ci_high = as.numeric(result_df[[upper_col[1]]][1])
  )
}


# ---------------------------------------------------------------------
# Per-mode analysis
# ---------------------------------------------------------------------

run_honestdid_for_mode <- function(config) {
  event_df <- load_event_study(config$coefficients)

  if (!(TARGET_EVENT_TIME %in% event_df$event_time)) {
    stop(
      paste0(
        "TARGET_EVENT_TIME = ", TARGET_EVENT_TIME,
        " is not available for ", config$label
      )
    )
  }

  if (TARGET_EVENT_TIME < 0) {
    stop("TARGET_EVENT_TIME must be a post-policy event time (>= 0).")
  }

  event_times <- event_df$event_time
  betahat <- event_df$beta
  sigma <- load_full_covariance(config$covariance, event_times)

  num_pre <- sum(event_times < 0)
  num_post <- sum(event_times >= 0)
  post_event_times <- event_times[event_times >= 0]

  # HonestDiD's l_vec indexes post-treatment coefficients only.
  target_post_index <- which(post_event_times == TARGET_EVENT_TIME)
  target_beta_index <- which(event_times == TARGET_EVENT_TIME)

  if (length(target_post_index) != 1 || length(target_beta_index) != 1) {
    stop(
      paste(
        "Could not uniquely locate the target effect for",
        config$label
      )
    )
  }

  l_vec <- rep(0, num_post)
  l_vec[target_post_index] <- 1

  # The estimate is read directly from the correctly indexed full coefficient
  # vector. It is not inferred from package-output column names.
  target_estimate <- betahat[target_beta_index]

  original_result <- HonestDiD::constructOriginalCS(
    betahat = betahat,
    sigma = sigma,
    numPrePeriods = num_pre,
    numPostPeriods = num_post,
    l_vec = l_vec
  )

  original_ci <- extract_ci(
    original_result,
    paste0(config$label, " original confidence set")
  )

  sensitivity_result <- HonestDiD::createSensitivityResults_relativeMagnitudes(
    betahat = betahat,
    sigma = sigma,
    numPrePeriods = num_pre,
    numPostPeriods = num_post,
    l_vec = l_vec,
    Mbarvec = MBAR_VALUES
  )

  sensitivity_df <- as.data.frame(sensitivity_result)

  if (!("Mbar" %in% names(sensitivity_df))) {
    stop(paste("HonestDiD output lacks Mbar for", config$label))
  }

  sensitivity_ci <- extract_ci(
    sensitivity_result,
    paste0(config$label, " sensitivity result")
  )

  # extract_ci returns the first row by design, so construct the full table
  # directly after validating the columns.
  lower_col <- intersect(
    c("lb", "lower", "Lower", "CI_Lower"),
    names(sensitivity_df)
  )[1]
  upper_col <- intersect(
    c("ub", "upper", "Upper", "CI_Upper"),
    names(sensitivity_df)
  )[1]

  summary_df <- bind_rows(
    data.frame(
      mode = config$label,
      target_event_time = TARGET_EVENT_TIME,
      estimand = paste0("Single-month event-study effect, event_time = ",
                        TARGET_EVENT_TIME),
      Mbar = 0,
      interval_type = "Original event-study CI",
      estimate = target_estimate,
      ci_low = original_ci$ci_low,
      ci_high = original_ci$ci_high
    ),
    data.frame(
      mode = config$label,
      target_event_time = TARGET_EVENT_TIME,
      estimand = paste0("Single-month event-study effect, event_time = ",
                        TARGET_EVENT_TIME),
      Mbar = as.numeric(sensitivity_df$Mbar),
      interval_type = "HonestDiD relative-magnitudes CI",
      estimate = target_estimate,
      ci_low = as.numeric(sensitivity_df[[lower_col]]),
      ci_high = as.numeric(sensitivity_df[[upper_col]])
    )
  ) %>%
    mutate(
      percent_effect = 100 * (exp(estimate) - 1),
      ci_low_percent = 100 * (exp(ci_low) - 1),
      ci_high_percent = 100 * (exp(ci_high) - 1),
      robust_negative = ci_high < 0,
      robust_positive = ci_low > 0
    )

  write_csv(summary_df, config$output_summary)

  plot <- ggplot(
    summary_df,
    aes(x = Mbar, y = percent_effect)
  ) +
    geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4) +
    geom_ribbon(
      aes(ymin = ci_low_percent, ymax = ci_high_percent),
      alpha = 0.20
    ) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2) +
    labs(
      title = paste0("HonestDiD sensitivity: ", config$label),
      subtitle = paste0(
        "Focal effect: event_time = ", TARGET_EVENT_TIME,
        ifelse(
          TARGET_EVENT_TIME == 1,
          " (February 2025, first fully treated month)",
          ""
        )
      ),
      x = "Allowed relative violation of parallel trends (Mbar)",
      y = "Percent effect",
      caption = paste(
        "Full event-study covariance matrix used.",
        "Bridge-and-tunnel estimates use aggregate treated and comparison facility groups."
      )
    ) +
    theme_minimal(base_size = 12)

  ggsave(
    filename = config$output_figure,
    plot = plot,
    width = 8,
    height = 5,
    dpi = 300
  )

  summary_df
}


# ---------------------------------------------------------------------
# Run all modes
# ---------------------------------------------------------------------

run_all_modes <- function() {
  all_results <- list()

  for (mode_name in names(MODE_CONFIG)) {
    config <- MODE_CONFIG[[mode_name]]

    message(strrep("=", 90))
    message("Running formal HonestDiD sensitivity: ", config$label)
    message(strrep("=", 90))

    all_results[[mode_name]] <- run_honestdid_for_mode(config)
  }

  combined <- bind_rows(all_results)
  write_csv(combined, OUT_COMBINED)

  con <- file(OUT_TXT, open = "wt", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)

  writeLines(strrep("=", 90), con)
  writeLines("12 Formal HonestDiD Sensitivity Results: All Modes", con)  
  writeLines(strrep("=", 90), con)
  writeLines("", con)
  writeLines(
    "This analysis uses the full variance-covariance matrix of the event-study",
    con
  )
  writeLines(
    "treatment-interaction coefficients. No diagonal covariance approximation is used.",
    con
  )
  writeLines("", con)
  writeLines(
    paste0(
      "Focal effect: event_time = ", TARGET_EVENT_TIME,
      ifelse(
        TARGET_EVENT_TIME == 1,
        " (February 2025, the first fully treated month).",
        "."
      )
    ),
    con
  )
  writeLines(
    "Mbar = 0 reports the original event-study confidence interval; positive Mbar",
    con
  )
  writeLines(
    "values report HonestDiD relative-magnitudes sensitivity intervals.",
    con
  )
  writeLines("", con)
  writeLines(
    "Scope: subway, bus, taxi, for-hire, and bridge-and-tunnel outcomes.",
    con
  )
  writeLines("", con)

  capture.output(
    print(
      combined %>%
        mutate(
          estimate = round(estimate, 4),
          ci_low = round(ci_low, 4),
          ci_high = round(ci_high, 4),
          percent_effect = round(percent_effect, 2),
          ci_low_percent = round(ci_low_percent, 2),
          ci_high_percent = round(ci_high_percent, 2)
        )
    ),
    file = con,
    append = TRUE
  )

  message(strrep("=", 90))
  message("Formal HonestDiD sensitivity complete")
  message("Saved: ", OUT_TXT)
  message("Saved: ", OUT_COMBINED)
  message(strrep("=", 90))

  invisible(combined)
}

run_all_modes()
