eval "$(conda shell.bash hook)"
source venv_oe/bin/activate

mkdir -p logs
bash einsum_ir/samples/tools/bench_opt_einsum.sh -l logs -r 100 -b torch -k all 2>&1 | tee logs/bench_opt_einsum.log

mkdir -p logs_ht
bash einsum_ir/samples/tools/bench_opt_einsum.sh -l logs_ht -r 100 -b torch -k all -p 1 2>&1 | tee logs_ht/bench_opt_einsum.log