Sapphire Rapids
===============
* us-east-2
* Fedora Cloud 41 AMI: Fedora-Cloud-Base-AmazonEC2.x86_64-41-1.4
(ami-09b3df5b58fafe605)
* c7i.metal-24xl
* 200 GiB EBS, gp3

Benchmarks (TVM)
----------------
mkdir logs
bash install_tvm.sh 2>&1 | tee logs/install_tvm.log
bash info.sh 2>&1 | tee logs/info.log
tmux
bash run_tvm.sh
# job ran tccg benchmarks, fctn, syn, tt and tw
# job crashed while running getd
# rescheduled with getd in the end
bash run_tvm_cont.sh
# aborted the TRN rerun as it took over 12h
# GETD rerun since it crash in verification.
# performed a second rerun of GETD with manually disabled verification
# (commenting https://github.com/scalable-analyses/einsum_ir/blob/56a6ea6a50901e760d55c1ecdd608937045f0289/samples/tools/tvm_helper.py#L213-L218)
tmux
source venv_tvm/bin/activate
TVM_LIBRARY_PATH=$(pwd)/tvm/build bash einsum_ir/samples/tools/bench_tvm.sh -k getd -c spr -n 1000 2>&1 | tee logs/bench_tvm_cont_2_getd.log

Benchmarks (einsum_ir)
----------------------
mkdir logs
bash install.sh 2>&1 | tee logs/install.log
tmux
bash run_einsum_ir.sh 2>&1 | tee logs/run_einsum_ir.log

Benchmarks (opt_einsum)
-----------------------
mkdir logs
bash install_opt_einsum.sh 2>&1 | tee logs/install_opt_einsum.log
tmux
bash run_opt_einsum.sh 2>&1 | tee logs/run_opt_einsum.log
# performed missing TBLIS optimization separately
mkdir logs
bash install.sh 2>&1 | tee logs/install_tblis_cont.log
tmux
export LD_LIBRARY_PATH=$(pwd)/venv_pytorch/lib/python3.10/site-packages/torch/lib/:$(pwd)/tblis/lib:$LD_LIBRARY_PATH
bash einsum_ir/samples/tools/bench_einsum_ir.sh -e einsum_ir/build_llvm/bench_expression -l logs -r 100 -b tblis -d 0 -k tccg_blocked_reordered  2>&1 | tee logs/run_einsum_ir_cont.log

Benchmarks (oneDNN)
-------------------
mkdir logs
# note that we installed the system packages on the instance before
bash install_blas.sh 2>&1 | tee logs/install_blas.log
tmux
bash run_blas.sh 2>&1 | tee logs/run_blas.log