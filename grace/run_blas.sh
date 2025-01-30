export LD_LIBRARY_PATH=$(pwd)/venv_pytorch/lib/python3.9/site-packages/torch/lib/:$(pwd)/tblis/lib:$LD_LIBRARY_PATH

mkdir -p logs/fc
cat /dev/null > logs/fc/nvpl.log

num_reps=100

echo "Running FC with NVPL"
echo "  num_reps: $num_reps"
date
echo -n "benchmarking"

for ((rep=1; rep<=num_reps; rep++))
do
  echo -n "."
  OMP_PROC_BIND=false OMP_NUM_THREADS=72 OMP_PLACES=cores ./blas/build/bench_gemm 2048 2048 2048 2048 2048 2048 sgemm 10 >> logs/fc/nvpl.log
done
echo ""
date
echo "Done"