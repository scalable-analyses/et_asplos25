echo "**********************************"
echo "*** Installing System Packages ***"
echo "**********************************"

sudo dnf install -y vim htop gfortran clang wget git cmake environment-modules g++-13 python3.10 llvm18-devel tmux zlib-devel libxml2-devel

echo "**********************"
echo "*** Installing TVM ***"
echo "**********************"

python3.10 -m venv venv_tvm
source venv_tvm/bin/activate

git clone --recursive https://github.com/apache/tvm tvm
cd tvm
git checkout v0.18.0
git submodule init
git submodule update
git log | head -n 25

mkdir build
cd build
cmake -DUSE_LLVM="llvm-config-18 --link-static" ..
make -j48
cd ..

pip install scipy psutil ml-dtypes xgboost torch
cd python
python setup.py install
cd ../..

git clone https://github.com/scalable-analyses/einsum_ir.git einsum_ir
cd einsum_ir
git log | head -n 25
cd ..