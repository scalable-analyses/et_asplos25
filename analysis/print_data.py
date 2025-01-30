import os
import pandas as pd
import glob

def process_data_directory(base_dir):
    results = []
    
    # Walk through machine directories
    for machine in ['grace', 'ryzen_8700g', 'spr']:
        machine_path = os.path.join(base_dir, machine)
        if not os.path.exists(machine_path):
            continue
            
        # Walk through benchmark directories
        for benchmark in ['fctn', 'getd', 'mera', 'syn', 'tnlm', 'trn', 'tt', 'tw']:
            bench_path = os.path.join(machine_path, benchmark)
            if not os.path.exists(bench_path):
                continue
                
            # Process all CSV files in the benchmark directory
            for csv_file in glob.glob(os.path.join(bench_path, '*.csv')):
                try:
                    df = pd.read_csv(csv_file)
                    if df.empty:
                        continue
                        
                    # Extract the first row (assuming one measurement per file)
                    row = df.iloc[0]
                    
                    # Determine which GFLOPS metric to use and round to integer
                    gflops = row['gflops_total_mean']
                    if pd.notnull(row.get('gflops_eval_mean')) and row['gflops_eval_mean'] > 0:
                        gflops = row['gflops_eval_mean']

                    results.append({
                        'machine': machine,
                        'benchmark': benchmark,
                        'code': row['code'],
                        'backend': row['backend'],
                        'gflops': gflops,
                        'compile_time': row['time_compile_mean']
                    })
                except Exception as e:
                    print(f"Error processing {csv_file}: {e}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort the results
    results_df = results_df.sort_values(['machine', 'benchmark', 'code', 'backend'])
    
    return results_df

if __name__ == "__main__":
    # Adjust this path to match your data directory location
    data_dir = "data"
    
    results = process_data_directory(data_dir)
    
    # Print summary
    print("\nPerformance Summary:")
    print("===================")
    
    # Group by machine, benchmark, code, and backend to show average performance
    summary = results.groupby(['machine', 'benchmark', 'code', 'backend']).agg({
        'gflops': lambda x: int(round(x)),
        'compile_time': 'mean'
    })
    
    print(summary)
    
    # Optionally save to CSV
    summary.to_csv('performance_summary.csv')