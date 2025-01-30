bash install_plot.sh
source venv_plot/bin/activate
bash summarize_all.sh
bash plot_all.sh
python print_data.py
python csv_to_latex.py performance_summary.csv
python compilation_time_to_latex.py performance_summary.csv grace

# SPR OneDNN GFLOPS
grep "perf,cpu" ../spr/logs/fc/onednn.log  | awk -F',' '
{
    values[NR] = $NF  # Store each value
    sum += $NF       # Add to sum
}
END {
    printf "Number of measurements: %d\n", NR
    printf "Average GFLOPS: %.2f\n", sum/NR
}'