eval "$(conda shell.bash hook)"
conda activate ./conda_oe

mkdir -p logs
bash einsum_ir/samples/tools/bench_opt_einsum.sh -l logs -r 100 -b torch -k all 2>&1 | tee logs/bench_opt_einsum.log