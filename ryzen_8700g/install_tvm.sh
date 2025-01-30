eval "$(conda shell.bash hook)"

echo "**********************"
echo "*** Installing TVM ***"
echo "**********************"

conda create -p $(pwd)/conda_tvm python=3.8 -y
conda activate ./conda_tvm
conda install -c conda-forge gcc=12.1.0 -y

git clone --recursive https://github.com/apache/tvm tvm
cd tvm
# git checkout e6bfaf8
git checkout v0.18.0
git submodule init
git submodule update
git log | head -n 25

mkdir build
cd build
cmake -DUSE_LLVM="llvm-config-18 --link-static" ..
make -j16
cd ..

pip install scipy psutil ml-dtypes xgboost torch
cd python
# ensure to remove possible local tvm installation in ~/.local
# before running the following command
python setup.py install
cd ../..

git clone https://github.com/scalable-analyses/einsum_ir.git einsum_ir
cd einsum_ir
git log | head -n 25
cd ..