export LD_LIBRARY_PATH=$(pwd)/venv_pytorch/lib/python3.10/site-packages/torch/lib/:$(pwd)/tblis/lib:$LD_LIBRARY_PATH

mkdir -p logs/fc
cat /dev/null > logs/fc/openblas.log

num_reps=100

echo "Running FC with OpenBLAS"
echo "  num_reps: $num_reps"
date
echo -n "benchmarking"

for ((rep=1; rep<=num_reps; rep++))
do
  echo -n "."
  OMP_NUM_THREADS=8 taskset -c 0,1,2,3,4,5,6,7 ./blas/build/bench_gemm 2048 2048 2048 2048 2048 2048 sgemm 10 >> logs/fc/openblas.log
done
echo ""
date
echo "Done"