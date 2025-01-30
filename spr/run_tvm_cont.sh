source venv_tvm/bin/activate

mkdir -p logs
TVM_LIBRARY_PATH=$(pwd)/tvm/build bash einsum_ir/samples/tools/bench_tvm.sh -k trn -c spr -n 1000 2>&1 | tee logs/bench_tvm_cont_trn.log

TVM_LIBRARY_PATH=$(pwd)/tvm/build bash einsum_ir/samples/tools/bench_tvm.sh -k getd -c spr -n 1000 2>&1 | tee logs/bench_tvm_cont_getd.log