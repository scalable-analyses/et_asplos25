export LD_LIBRARY_PATH=$(pwd)/venv_pytorch/lib/python3.10/site-packages/torch/lib/:$(pwd)/tblis/lib:$LD_LIBRARY_PATH
bash einsum_ir/samples/tools/bench_einsum_ir.sh -e einsum_ir/build_llvm/bench_expression -l logs -r 100 -b tpp -d 1 -k syn,tt,fctn,tw,getd,trn,mera,tnlm,tccg_blocked

bash einsum_ir/samples/tools/bench_einsum_ir.sh -e einsum_ir/build_llvm/bench_expression -l logs -r 100 -b tpp -d 0 -k tccg_blocked_reordered,fc

bash einsum_ir/samples/tools/bench_einsum_ir.sh -e einsum_ir/build_llvm/bench_expression -l logs -r 100 -b blas -d 1 -k syn,tt,fctn,tw,getd,trn,mera,tnlm,tccg_blocked

bash einsum_ir/samples/tools/bench_einsum_ir.sh -e einsum_ir/build_llvm/bench_expression -l logs -r 100 -b blas -d 0 -k tccg_blocked_reordered,fc

bash einsum_ir/samples/tools/bench_einsum_ir.sh -e einsum_ir/build_llvm/bench_expression -l logs -r 100 -b tblis -d 1 -k all


bash einsum_ir/samples/tools/bench_einsum_ir.sh -e einsum_ir/build_llvm/bench_expression -l logs_ht -r 100 -b tpp -d 1 -k syn,tt,fctn,tw,getd,trn,mera,tnlm,tccg_blocked -p 1

bash einsum_ir/samples/tools/bench_einsum_ir.sh -e einsum_ir/build_llvm/bench_expression -l logs_ht -r 100 -b tpp -d 0 -k tccg_blocked_reordered,fc -p 1

bash einsum_ir/samples/tools/bench_einsum_ir.sh -e einsum_ir/build_llvm/bench_expression -l logs_ht -r 100 -b blas -d 1 -k syn,tt,fctn,tw,getd,trn,mera,tnlm,tccg_blocked -p 1

bash einsum_ir/samples/tools/bench_einsum_ir.sh -e einsum_ir/build_llvm/bench_expression -l logs_ht -r 100 -b blas -d 0 -k tccg_blocked_reordered,fc -p 1

bash einsum_ir/samples/tools/bench_einsum_ir.sh -e einsum_ir/build_llvm/bench_expression -l logs_ht -r 100 -b tblis -d 1 -k all -p 1