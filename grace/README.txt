Grace
=====
* NVIDIA Grace CPU Superchip (https://resources.nvidia.com/en-us-grace-cpu/data-center-datasheet)
    * Core count: 144 Arm Neoverse V2 Cores with 4x128b SVE2
    * L1 cache: 64KB i-cache + 64KB d-cache
    * L2 cache: 1MB per core
    * L3: cache: 228MB
    * Base Frequency: 3.1 GHz
    * All-Core SIMD Frequency: 3.0 GHz
    * LPDDR5X size: 240GB, 480GB, and 960GB on-module memory options
    * Memory bandwidth: Up to 768 GB/s for 960GB Up to 1024 GB/s for 240GB, 480GB
    * NVIDIA NVLink-C2C bandwidth: 900GB/s
    * PCIe links: Up to 8x PCIe Gen5 x16 option to bifurcate
    * Module thermal design power (TDP): 500W TDP with memory
    * Form factor: Superchip module
    * Thermal solution: Air cooled or liquid cooled
* 480GB Co-Packaged LPDDR5X-4800MHz w/ECC

Disable NUMA Balancing
----------------------
sudo sysctl kernel.numa_balancing=0
cat /proc/sys/kernel/numa_balancing

Disable Second Socket
---------------------
sudo bash disable_2nd_socket.sh

NVPL Installation
-----------------
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/sbsa/cuda-rhel9.repo
sudo dnf clean all
sudo dnf install nvpl

Benchmarks (TVM)
----------------
mkdir logs
bash install_tvm.sh 2>&1 | tee logs/install_tvm.log
bash info.sh 2>&1 | tee logs/info.log
tmux
bash run_tvm.sh

Benchmarks (einsum_ir)
----------------------
mkdir logs
bash install.sh 2>&1 | tee logs/install.log
tmux
bash run_einsum_ir.sh 2>&1 | tee logs/run_einsum_ir.log
# aborted the HT runs since they were slow
# performed missing TBLIS optimization separately
mkdir logs
bash install.sh 2>&1 | tee logs/install_tblis_cont.log
tmux
export LD_LIBRARY_PATH=$(pwd)/venv_pytorch/lib/python3.9/site-packages/torch/lib/:$(pwd)/tblis/lib:$LD_LIBRARY_PATH
bash einsum_ir/samples/tools/bench_einsum_ir.sh -e einsum_ir/build_llvm/bench_expression -l logs -r 100 -b tblis -d 0 -k tccg_blocked_reordered  2>&1 | tee logs/run_einsum_ir_cont.log

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