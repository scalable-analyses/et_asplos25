import csv
import pandas as pd
import argparse

def get_max_values_per_machine(df, bench, machines):
    """Returns dict with max gflops per machine for given benchmark"""
    max_values = {}
    for machine in machines:
        # Exclude 'blas' backend from max calculation
        mask = (df['machine'] == machine) & \
               (df['benchmark'] == bench.lower()) & \
               (df['backend'] != 'blas')  # Add this condition
        values = df[mask]['gflops'].values
        if len(values) > 0:
            max_values[machine] = max(values)
        else:
            max_values[machine] = 0
    return max_values

def get_speedup(df, bench, machines):
    """Calculate speedup of ET-TPP over best of other approaches"""
    speedups = {}
    for machine in machines:
        # Get ET-TPP performance
        tpp_mask = (df['machine'] == machine) & \
                  (df['benchmark'] == bench.lower()) & \
                  (df['code'] == 'einsum_ir') & \
                  (df['backend'] == 'tpp')
        tpp_value = df[tpp_mask]['gflops'].values

        # Get max of other approaches (excluding TPP and blas)
        others_mask = (df['machine'] == machine) & \
                     (df['benchmark'] == bench.lower()) & \
                     ~((df['code'] == 'einsum_ir') & (df['backend'] == 'tpp')) & \
                     (df['backend'] != 'blas')
        others_values = df[others_mask]['gflops'].values
        
        if len(tpp_value) > 0 and len(others_values) > 0:
            speedups[machine] = tpp_value[0] / max(others_values)
        else:
            speedups[machine] = 0
    return speedups

# Parse command line arguments
parser = argparse.ArgumentParser(description='Convert CSV to LaTeX table')
parser.add_argument('csv_file', help='Path to the input CSV file')
args = parser.parse_args()

# Read the CSV file
df = pd.read_csv(args.csv_file)

# Define mapping from CSV identifiers to LaTeX table identifiers
backend_mapping = {
    ('einsum_ir', 'tpp'): 'ET-TPP',
    ('einsum_ir', 'tblis'): 'ET-TBLIS',
    ('tvm', 'ansor'): 'TVM-Ansor',
    ('at::einsum', 'eigen'): 'ATen',
    ('opt_einsum', 'torch'): 'OE-Torch'
}

# Define benchmark order as shown in the example
benchmark_order = ['SYN', 'TT', 'FCTN', 'TW', 'GETD', 'TRN', 'MERA', 'TNLM']
benchmark_mapping = {
    'syn': 'SYN',
    'tt': 'TT',
    'fctn': 'FCTN',
    'tw': 'TW',
    'getd': 'GETD',
    'trn': 'TRN',
    'mera': 'MERA',
    'tnlm': 'TNLM'
}

# Start LaTeX table
print(r"""\begin{table}
  \begin{center}
    \caption{Sustained FP32 GFLOPS for the benchmarked contraction trees on the three testbeds. Speedup gives the performance improvement of ET-TPP over the other approaches.}
    \label{tab:einsum_perf}
    \begin{tabular}{|c|c|c|c|c|}
      \hline
      ID & Approach & Grace & SPR & Ryzen \\
      \hline""")

# Process each benchmark
for bench in benchmark_order:
    print(f"      \\multirow{{6}}{{*}}{{{bench}}}")
    
    # Get max values for this benchmark
    machines = ['grace', 'spr', 'ryzen_8700g']
    max_values = get_max_values_per_machine(df, bench.lower(), machines)
    
    # For each backend type
    for (code, backend), latex_name in backend_mapping.items():
        values = []
        for machine in machines:
            mask = (df['machine'] == machine) & \
                  (df['benchmark'] == bench.lower()) & \
                  (df['code'] == code) & \
                  (df['backend'] == backend)
            value = df[mask]['gflops'].values
            if len(value) > 0:
                val_str = str(int(value[0]))
                # Bold if it's the max value for this machine (exact match)
                if abs(value[0] - max_values[machine]) < 0.1:  # Much smaller tolerance
                    val_str = f"\\textbf{{{val_str}}}"
                values.append(val_str)
            else:
                values.append('N/A')
        
        # Print row with proper indentation
        print(f"        & {latex_name:<10} & {values[0]:>4} & {values[1]:>4} & {values[2]:>4} \\\\")

    # Add speedup row
    speedups = get_speedup(df, bench, machines)
    speedup_values = []
    for machine in machines:
        if speedups[machine] > 0:
            speedup_str = f"{speedups[machine]:.1f}$\\times$"
            if speedups[machine] > 1.0:
                speedup_str = f"\\textbf{{{speedup_str}}}"
            speedup_values.append(speedup_str)
        else:
            speedup_values.append("N/A")
    
    print(f"        \\cline{{2-5}}")
    print(f"        & Speedup & {speedup_values[0]} & {speedup_values[1]} & {speedup_values[2]} \\\\")
    print("      \\hline")

# End LaTeX table
print(r"""    \end{tabular}
  \end{center}
\end{table}""")