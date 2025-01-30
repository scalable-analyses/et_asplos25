eval "$(conda shell.bash hook)"

echo "**********************"
echo "*** Installing TVM ***"
echo "**********************"

conda create -p ./conda_tvm python=3.8 -y
conda activate ./conda_tvm

git clone https://github.com/apache/tvm tvm
cd tvm
git checkout v0.18.0
git submodule init
git submodule update
git log | head -n 25

mkdir build
cd build
cmake -DUSE_LLVM=ON ..
make -j
cd ..

pip install scipy psutil ml-dtypes xgboost torch
cd python
python setup.py install
cd ../..

git clone https://github.com/scalable-analyses/einsum_ir.git einsum_ir
cd einsum_ir
git log | head -n 25
cd ..