source venv_tvm/bin/activate

mkdir -p logs
TVM_LIBRARY_PATH=$(pwd)/tvm/build bash einsum_ir/samples/tools/bench_tvm.sh -k tccg_default,tccg_blocked_reordered,syn,tt,fctn,tw,getd,trn -c spr -n 1000 2>&1 | tee logs/bench_tvm.log