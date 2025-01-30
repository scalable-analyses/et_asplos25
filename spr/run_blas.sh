mkdir logs/fc
cat /dev/null > logs/fc/onednn.log

num_reps=100

echo "Running FC with oneDNN"
echo "  num_reps: $num_reps"
date
echo -n "benchmarking"

for ((rep=1; rep<=num_reps; rep++))
do
  echo -n "."
  OMP_NUM_THREADS=96 OMP_PLACES={0}:96:1 ./oneDNN/build/tests/benchdnn/benchdnn --matmul --mode=P --dt=f32 2048x2048:2048x2048 >> logs/fc/onednn.log
done
echo ""
date
echo "Done"