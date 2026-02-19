library(tidyverse)
library(ggh4x)

# --- Data loading and preparation ---

# Station name lookup
station_lookup <- read_tsv("data_in/trenderBVVF_yearly.txt",
                           locale = locale(encoding = "UTF-8")) %>%
  distinct(REG_ID, STATN) %>%
  deframe()

trends <- read_csv("data_out/trends3.csv") %>%
  rename(
    REG_ID = `Provplats ID (Stationsregistret)`,
    parameter = Mätvariabel,
    year = År,
    observed = Årsvärde,
    fitted = `Trendvärde (modellvärde)`,
    trend = `Trend (modelltrend)`
  ) %>%
  mutate(STATN = station_lookup[as.character(REG_ID)])

# Station order (north to south)
station_order <- c(
  "KOSTERFJORDEN NR16", "BJÖRKHOLMEN", "BYTTELOCKET", "STRETUDDEN",
  "BYFJORDEN", "HAVSTENSFJORD", "KOLJÖFJORD", "GALTERÖ",
  "ÅSTOL", "INSTÖ RÄNNA", "E ÄLVSBORGSBRON", "SKALKORGARNA",
  "DANAFJORD", "VALÖ"
)

trends <- trends %>%
  mutate(STATN = factor(STATN, levels = rev(station_order)))

# Trend color mapping
trend_colors <- c(
  "minskande" = "#4A90D9",
  "neutral"   = "#F5C542",
  "\u00f6kande"    = "#D94A4A"
)

trend_labels <- c(
  "minskande" = "Nedåtgående trend",
  "neutral"   = "Ingen signifikant trend",
  "\u00f6kande"    = "Uppåtgående trend"
)

# --- Surface parameters plot (12 facets) ---

surface_params <- c(
  "NO2 0-10 m - hel\u00e5r",  "NO3 0-10 m - hel\u00e5r",
  "NH4 0-10 m - hel\u00e5r",  "DIN 0-10 m - hel\u00e5r",
  "TOTN 0-10 m - hel\u00e5r", "PO4 0-10 m - hel\u00e5r",
  "TOTP 0-10 m - hel\u00e5r", "SiO3 0-10 m - hel\u00e5r",
  "SECCHI 0-10 m - hel\u00e5r", "TEMPERATUR 0-10 m - hel\u00e5r",
  "O2 0-10 m - hel\u00e5r",  "CHL a 0-10 m - hel\u00e5r"
)

# Short facet labels with subscript formatting
facet_labels <- c(
  "NO2 0-10 m - hel\u00e5r"          = "NO\u2082",
  "NO3 0-10 m - hel\u00e5r"          = "NO\u2083",
  "NH4 0-10 m - hel\u00e5r"          = "NH\u2084",
  "DIN 0-10 m - hel\u00e5r"          = "DIN",
  "TOTN 0-10 m - hel\u00e5r"         = "TOTN",
  "PO4 0-10 m - hel\u00e5r"          = "PO\u2084",
  "TOTP 0-10 m - hel\u00e5r"         = "TOTP",
  "SiO3 0-10 m - hel\u00e5r"         = "SiO\u2083",
  "SECCHI 0-10 m - hel\u00e5r"       = "SECCHI",
  "TEMPERATUR 0-10 m - hel\u00e5r"   = "TEMPERATUR",
  "O2 0-10 m - hel\u00e5r"           = "O\u2082",
  "CHL a 0-10 m - hel\u00e5r"        = "CHL a"
)

surface_data <- trends %>%
  filter(parameter %in% surface_params) %>%
  mutate(parameter = factor(parameter, levels = surface_params))

p_surface <- ggplot(surface_data, aes(x = year, y = STATN, fill = trend)) +
  geom_tile(height = 0.85, width = 1) +
  facet_wrap(~ parameter, ncol = 4, labeller = labeller(parameter = facet_labels)) +
  scale_fill_manual(
    values = trend_colors,
    labels = trend_labels,
    na.value = "white"
  ) +
  scale_x_continuous(
    breaks = seq(1990, 2020, by = 10),
    expand = c(0, 0)
  ) +
  labs(
    title = "Signifikanta trender i ytvattnet, 0\u201310 m (GAM)",
    x = NULL, y = NULL, fill = "Signifikanta trender"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    strip.text = element_text(face = "bold", size = 11),
    axis.text.y = element_text(size = 7),
    axis.text.x = element_text(size = 8),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    panel.grid = element_blank(),
    panel.spacing = unit(0.8, "lines"),
    plot.title = element_text(face = "bold", size = 13, hjust = 0.5)
  )

ggsave("plots/BVVF_surface_trends.png", p_surface,
       width = 18, height = 14, dpi = 200, bg = "white")
message("Saved: plots/BVVF_surface_trends.png")

# --- Bottom water O2 plot (helår + höst faceted) ---

bw_params <- c(
  "O2 Bottenvatten - hel\u00e5r",
  "O2 Bottenvatten - h\u00f6st"
)

bw_labels <- c(
  "O2 Bottenvatten - hel\u00e5r" = "O\u2082 Bottenvatten \u2013 hel\u00e5r",
  "O2 Bottenvatten - h\u00f6st"  = "O\u2082 Bottenvatten \u2013 h\u00f6st (aug\u2013okt)"
)

bw_data <- trends %>%
  filter(parameter %in% bw_params) %>%
  mutate(parameter = factor(parameter, levels = bw_params))

p_bw <- ggplot(bw_data, aes(x = year, y = STATN, fill = trend)) +
  geom_tile(height = 0.85, width = 1) +
  facet_wrap(~ parameter, ncol = 2, labeller = labeller(parameter = bw_labels)) +
  scale_fill_manual(
    values = trend_colors,
    labels = trend_labels,
    na.value = "white"
  ) +
  scale_x_continuous(
    breaks = seq(1990, 2020, by = 10),
    expand = c(0, 0)
  ) +
  labs(
    title = "Signifikanta trender i bottenvatten \u2013 syrgaskoncentration (GAM)",
    x = NULL, y = NULL, fill = "Signifikanta trender"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    strip.text = element_text(face = "bold", size = 11),
    axis.text.y = element_text(size = 8),
    axis.text.x = element_text(size = 9),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    panel.grid = element_blank(),
    panel.spacing = unit(1.2, "lines"),
    plot.title = element_text(face = "bold", size = 13, hjust = 0.5)
  )

ggsave("plots/BVVF_bottom_O2_trends.png", p_bw,
       width = 14, height = 6, dpi = 200, bg = "white")
message("Saved: plots/BVVF_bottom_O2_trends.png")

# --- Time series plots: observed values + GAM trend line ---

# Re-level STATN for time series (top to bottom = north to south)
trends_ts <- trends %>%
  mutate(STATN = factor(STATN, levels = station_order))

# Helper: build a time series facet_grid plot
plot_timeseries <- function(data, param_vec, label_vec, title, filename,
                            width, height, ncol_params = 4) {

  ts_data <- data %>%
    filter(parameter %in% param_vec) %>%
    mutate(
      param_label = factor(label_vec[as.character(parameter)],
                           levels = label_vec[param_vec])
    )

  p <- ggplot(ts_data, aes(x = year)) +
    geom_point(aes(y = observed), size = 0.6, alpha = 0.5, color = "grey40") +
    geom_line(aes(y = fitted, color = trend), linewidth = 0.8) +
    scale_color_manual(
      values = trend_colors,
      labels = trend_labels,
      na.value = "grey70"
    ) +
    facet_grid2(
      STATN ~ param_label,
      scales = "free_y",
      independent = "y"
    ) +
    scale_x_continuous(breaks = seq(1990, 2020, by = 10)) +
    labs(title = title, x = NULL, y = NULL, color = "Trend") +
    theme_minimal(base_size = 9) +
    theme(
      strip.text.x = element_text(face = "bold", size = 10),
      strip.text.y = element_text(size = 7, angle = 0, hjust = 0),
      axis.text.y = element_text(size = 6),
      axis.text.x = element_text(size = 7, angle = 45, hjust = 1),
      legend.position = "bottom",
      legend.title = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.spacing = unit(0.3, "lines"),
      plot.title = element_text(face = "bold", size = 13, hjust = 0.5)
    )

  ggsave(filename, p, width = width, height = height, dpi = 200, bg = "white")
  message("Saved: ", filename)
}

# Surface time series: split into 3 rows of 4 parameters to keep readable
plot_timeseries(
  trends_ts,
  param_vec = surface_params,
  label_vec = facet_labels,
  title = "Observerade v\u00e4rden och GAM-trend, ytvatten 0\u201310 m",
  filename = "plots/BVVF_surface_timeseries.png",
  width = 20, height = 22
)

# Bottom O2 time series
bw_ts_labels <- c(
  "O2 Bottenvatten - hel\u00e5r" = "Hel\u00e5r",
  "O2 Bottenvatten - h\u00f6st"  = "H\u00f6st (aug\u2013okt)"
)

plot_timeseries(
  trends_ts,
  param_vec = bw_params,
  label_vec = bw_ts_labels,
  title = "Observerade v\u00e4rden och GAM-trend, bottenvatten O\u2082",
  filename = "plots/BVVF_bottom_O2_timeseries.png",
  width = 10, height = 16
)
