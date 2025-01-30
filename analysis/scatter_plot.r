library(ggplot2)

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) > 0) args[1] else "data/grace/tccg_blocked_reordered"
output_file <- if (length(args) > 1) args[2] else "tccg_grace.pdf"

# Read individual CSV files
data_ansor <- read.csv(file.path(data_dir, "tvm_backend_ansor.csv"))
data_torch <- read.csv(file.path(data_dir, "opt_einsum_backend_torch.csv"))
data_tpp <- read.csv(file.path(data_dir, "einsum_ir_backend_tpp.csv"))
data_tblis <- read.csv(file.path(data_dir, "einsum_ir_backend_tblis.csv"))
data_eigen <- read.csv(file.path(data_dir, "aten_backend_eigen.csv"))

# Combine all data frames
data_list <- list(
  transform(data_ansor, backend="ansor"),
  transform(data_torch, backend="torch"),
  transform(data_tpp, backend="tpp"),
  transform(data_tblis, backend="tblis"),
  transform(data_eigen, backend="eigen")
)

data <- do.call(rbind, data_list)

# Replace -9999 values with 0
data[data == -9999] <- 0

backend_order <- c("tpp", "tblis", "ansor", "eigen", "torch")
backend_labels <- c( "ET-TPP", "ET-TBLIS", "TVM-Ansor", "ATen", "OE-Torch")
data$backend <- factor(data$backend, levels = backend_order, labels = backend_labels)

# Sort the dataframe so TPP appears last
data <- data[order(match(data$backend, backend_order)), ]

# For each backend, create sequential IDs (1-24) based on order of appearance
data$id <- ave(seq_along(data$code), data$backend, FUN = seq_along)

# Create a column for plotting that uses different GFLOPS measures based on backend
data$gflops_plot <- ifelse(data$code == "at::einsum",
                          data$gflops_total_mean,
                          data$gflops_eval_mean)

# Create the scatter plot
p <- ggplot(data, aes(x=id, y=gflops_plot, color=backend, shape=backend)) +
  geom_point(size=3) +  # Larger points for better visibility
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    plot.title = element_text(hjust = 0.5),
    legend.position = "bottom",
    legend.key.size = unit(0.5, "cm"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  ) +
  scale_x_continuous(breaks = 1:24) +
  scale_y_continuous(breaks = seq(0, max(data$gflops_plot), by = ifelse(max(data$gflops_plot) < 1500, 200, 1000))) +
  scale_color_manual(values = c(
    "#E69F00",  # orange
    "#CC79A7",  # pink
    "#56B4E9",  # light blue
    "#009E73",  # green
    "#0072B2"   # dark blue
  )) +
  # Different shapes for each backend
  scale_shape_manual(values = c(8, 16, 18, 15, 17)) +  # different symbols
  labs(
    x = "TCCG ID",
    y = "GFLOPS",
    color = "",
    shape = ""
  )

ggsave(output_file, p, width = 6, height = 3)