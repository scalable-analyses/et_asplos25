Murray
======
* AMD Ryzen 7 8700G (https://www.amd.com/en/products/processors/desktops/ryzen/8000-series/amd-ryzen-7-8700g.html)
    * Architecture: Zen 4
    * # of CPU Cores: 8
    * Multithreading (SMT): Yes
    * # of Threads: 16
    * Max. Boost Clock: Up to 5.1 GHz
    * Base Clock: 4.2 GHz
    * L2 Cache: 8 MB
    * L3 Cache: 16 MB
    * Default TDP: 65W
    * AMD Configurable TDP (cTDP): 45-65W
    * Processor Technology for CPU Cores: TSMC 4nm FinFET
    * CPU Compute Die (CCD) Size: 178mm²
    * Package Die Count: 1
    * Supported Extensions: AES , AMD-V , AVX , AVX2 , AVX512 , FMA3 , MMX-plus , SHA , SSE , SSE2 , SSE3 , SSE4.1 , SSE4.2 , SSE4A , SSSE3 , x86-64
    * Launch Date: 1/31/2024
* 64 GiB DDR5-5200
* Ubuntu 22.04.4 LTS (GNU/Linux 6.8.7+ x86_64)
* Samsung 990 PRO (1TB)

System Software
---------------
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 18

sudo apt install libzstd-dev
sudo apt install libpolly-18-dev

Benchmarks (TVM)
----------------
tmux
mkdir logs
bash info.sh 2>&1 | tee logs/info.log
bash install_tvm.sh 2>&1 | tee logs/install_tvm.log
bash run_tvm.sh
# manually aborted TRN optimization due to missing progress
# performed rerun of GETD with manually disabled verification
# (commenting https://github.com/scalable-analyses/einsum_ir/blob/56a6ea6a50901e760d55c1ecdd608937045f0289/samples/tools/tvm_helper.py#L213-L218
# and setting rel_error=0)
tmux
eval "$(conda shell.bash hook)"
conda activate ./conda_tvm
mkdir -p logs
TVM_LIBRARY_PATH=$(pwd)/tvm/build bash einsum_ir/samples/tools/bench_tvm.sh -k getd -c zen4 -n 1000 2>&1 | tee logs/bench_tvm_getd.log

Benchmarks (einsum_ir)
----------------------
mkdir logs
bash install.sh 2>&1 | tee logs/install.log
tmux
bash run_einsum_ir.sh 2>&1 | tee logs/run_einsum_ir.log
# performed missing TBLIS optimization separately
mkdir logs
bash install.sh 2>&1 | tee logs/install_tblis_cont.log
tmux
export LD_LIBRARY_PATH=$(pwd)/venv_pytorch/lib/python3.10/site-packages/torch/lib/:$(pwd)/tblis/lib:$LD_LIBRARY_PATH
bash einsum_ir/samples/tools/bench_einsum_ir.sh -e einsum_ir/build_llvm/bench_expression -l logs -r 100 -b tblis -d 0 -k tccg_blocked_reordered 2>&1 | tee logs/run_einsum_ir_cont.log

Benchmarks (opt_einsum)
-----------------------
mkdir logs
bash install_opt_einsum.sh 2>&1 | tee logs/install_opt_einsum.log
tmux
bash run_opt_einsum.sh 2>&1 | tee logs/run_opt_einsum.log

Benchmarks (BLAS)
-----------------
mkdir logs
bash install_blas.sh 2>&1 | tee logs/install_blas.log
tmux
bash run_blas.sh 2>&1 | tee logs/run_blas.log