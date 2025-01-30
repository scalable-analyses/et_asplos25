import pandas as pd
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Convert compilation times to LaTeX table')
parser.add_argument('csv_file', help='Path to the input CSV file')
parser.add_argument('machine', help='Machine name to show compilation times for', 
                   choices=['grace', 'spr', 'ryzen_8700g'])
args = parser.parse_args()

# Read the CSV file
df = pd.read_csv(args.csv_file)

# Define backends to show
compile_backends = {
    ('einsum_ir', 'tpp'): 'ET-TPP',
    ('einsum_ir', 'tblis'): 'ET-TBLIS',
    ('tvm', 'ansor'): 'TVM-Ansor',
    ('opt_einsum', 'torch'): 'OE-Torch'
}

# Define benchmark order
benchmark_order = ['SYN', 'TT', 'FCTN', 'TW', 'GETD', 'TRN', 'MERA', 'TNLM']

# Start LaTeX table
machine_name = {'grace': 'Grace', 'spr': 'SPR', 'ryzen_8700g': 'Ryzen'}[args.machine]
print(r"""\begin{table}[htb!]
  \begin{center}
    \caption{Compilation time for the einsum tree approach using tensor processing primitives """ + \
    f"(ET-TPP) and TBLIS (ET-TBLIS), TVM Ansor (TVM), and opt\\_einsum (OE). " + \
    f"All measurements were performed on the {machine_name} system." + \
    r"""}
    \label{tab:compilation_time}
    \begin{tabular}{|c|r|r|r|r|}
      \hline
      ID & ET-TPP & ET-TBLIS & TVM-Ansor & OE-Torch \\
      \hline""")

# Process each benchmark
for bench in benchmark_order:
    values = []
    for (code, backend), _ in compile_backends.items():
        mask = (df['machine'] == args.machine) & \
               (df['benchmark'] == bench.lower()) & \
               (df['code'] == code) & \
               (df['backend'] == backend)
        time = df[mask]['compile_time'].values
        if len(time) > 0:
            if backend == 'ansor':
                # Convert seconds to minutes for TVM-Ansor with 2 decimals
                values.append(f"{time[0]/60:.2f}\\,m")
            else:
                # Convert seconds to milliseconds for others with 2 decimals
                values.append(f"{time[0] * 1000:.2f}\\,ms")
        else:
            values.append('--')
    
    print(f"      {bench:<4} & {values[0]:>9} & {values[1]:>9} & {values[2]:>8} & {values[3]:>9} \\\\")

print(r"""      \hline
    \end{tabular}
  \end{center}
\end{table}""")