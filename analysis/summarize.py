import pandas
import io
import argparse
import os, sys

def summarize(fh, code, backend, summary):
    lines = fh.readlines()

    if code == "tvm":
        lines_code = [line for line in lines if line.startswith("CSV_DATA:")]
    else:
      lines_code = [line for line in lines if line.startswith("CSV_DATA: " + code)]

    # Modified parsing to handle both with and without space after CSV_DATA:
    if code == "tvm":
        lines_code = [line[len("CSV_DATA:"):].strip() for line in lines_code]
    else:
        lines_code = [line[len("CSV_DATA: " + code + ','):] for line in lines_code]
    
    lines_code = [line.replace("\n", "") for line in lines_code]

    # Different column handling based on code type
    if code == "sgemm":
        df = pandas.read_csv(io.StringIO("\n".join(lines_code)),
                           header=None,
                           names=["m", "n", "k", "lda", "ldb", "ldc", 
                                 "num_flops", "time_total", "gflops_total"])
        df["einsum_string"] = "ab,ca->cb"
        df["dim_sizes"] = df["m"].astype(str) + "," + df["n"].astype(str) + "," + df["k"].astype(str)
        df["time_compile"] = 0
        df["time_eval"] = 0
        df["gflops_eval"] = df["gflops_total"]
    elif code == "opt_einsum":
        df = pandas.read_csv(io.StringIO("\n".join(lines_code)),
                            header=None,
                            names=["backend", "einsum_string", "dim_sizes", "cont_path", 
                                  "dtype", "num_flops", "arithmetic_intensity",
                                  "time_compile", "time_eval", "gflops_eval", "gflops_total"])
    elif code == "tvm":
        df = pandas.read_csv(io.StringIO("\n".join(lines_code)),
                            header=None, 
                            names=["einsum_string", "dim_sizes", "dtype", "hardware_params",
                                  "target", "num_trials", "timeout", "time_compile",
                                  "rel_error", "num_flops", "time_eval", "gflops_eval"])
        # Add gflops_total calculation for consistency
        df["gflops_total"] = df["num_flops"] / ((df["time_compile"] + df["time_eval"]) * 1e9)
    else:
        # Original column handling for einsum_ir etc
        df = pandas.read_csv(io.StringIO("\n".join(lines_code)),
                            header=None,
                            names=["einsum_string", "dim_sizes", "cont_path", "num_flops",
                                  "time_compile", "time_eval", "gflops_eval", "gflops_total"])

    # derive list of keys containing unique (einsum string, dim_sizes) pairs
    keys = df[["einsum_string", "dim_sizes"]].drop_duplicates()

    for key in keys.iterrows():
        key = key[1]
        key_df = df[(df["einsum_string"] == key["einsum_string"]) & (df["dim_sizes"] == key["dim_sizes"])]
        key_summary = key_df[["einsum_string", "dim_sizes", "num_flops"]].iloc[0]
        # code
        key_summary["code"] = code
        # backend
        key_summary["backend"] = backend
        # mean
        key_summary["time_compile_mean"] = key_df["time_compile"].mean()
        key_summary["time_eval_mean"] = key_df["time_eval"].mean()
        key_summary["gflops_eval_mean"] = key_df["gflops_eval"].mean()
        key_summary["gflops_total_mean"] = key_df["gflops_total"].mean()
        # std deviation
        key_summary["time_compile_std_dev"] = key_df["time_compile"].std()
        key_summary["time_eval_std_dev"] = key_df["time_eval"].std()
        key_summary["gflops_eval_std_dev"] = key_df["gflops_eval"].std()
        key_summary["gflops_total_std_dev"] = key_df["gflops_total"].std()

        # number of samples
        key_summary["num_samples"] = key_df.shape[0]

        # add row
        summary = summary._append(key_summary)

    return summary

def main():
    parser = argparse.ArgumentParser(description='Summarize einsum benchmark results')
    parser.add_argument('--input', 
                      help='Input log file path',
                      required=True)
    parser.add_argument('--output',
                      help='Output CSV file path',
                      required=True)
    parser.add_argument('--code',
                      help='Code type (e.g., einsum_ir, at::einsum, tvm)',
                      required=True)
    parser.add_argument('--backend',
                      help='Backend type (e.g., tpp, blas, tblis, eigen)',
                      required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    # Initialize empty summary dataframe
    summary = pandas.DataFrame(columns=["code",
                                      "backend", 
                                      "einsum_string",
                                      "dim_sizes",
                                      "cont_path",
                                      "num_flops",
                                      "time_compile_mean",
                                      "time_eval_mean",
                                      "gflops_eval_mean",
                                      "gflops_total_mean",
                                      "time_compile_std_dev",
                                      "time_eval_std_dev", 
                                      "gflops_eval_std_dev",
                                      "gflops_total_std_dev",
                                      "num_samples"])

    # Process single input file
    with open(args.input, 'r') as fh:
        summary = summarize(fh,
                          args.code,
                          args.backend, 
                          summary)

    # Write to output CSV
    summary.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()