eval "$(conda shell.bash hook)"

echo "**************************"
echo "*** Installing Pytorch ***"
echo "**************************"

python -m venv venv_pytorch
source venv_pytorch/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__config__.show()); print(torch.__config__.parallel_info());"

echo "*****************************************"
echo "*** Installing Einsum IR Dependencies ***"
echo "*****************************************"

git clone https://github.com/libxsmm/libxsmm.git
cd libxsmm
git log | head -n 25
make BLAS=0 -j
cd ..

wget https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.28.tar.gz
tar -xvf v0.3.28.tar.gz
cd OpenBLAS-0.3.28
make -j
make PREFIX=$(pwd)/../openblas install
cd ..

git clone https://github.com/devinamatthews/tblis.git tblis_src
cd tblis_src
git checkout 2cbdd21
git log | head -n 25
./configure --prefix=$(pwd)/../tblis --enable-thread-model=openmp
sed -i '971d' Makefile
make -j
make install
cd ..
rm -rf tblis_src

for dir_type in indexed_dpd indexed dpd fwd
do
mkdir -p tblis/include/tblis/external/marray/marray/${dir_type}
mv tblis/include/tblis/external/marray/marray/*${dir_type}*.hpp tblis/include/tblis/external/marray/marray/${dir_type}
done

mkdir -p tblis/include/tblis/external/marray/marray/detail
mv tblis/include/tblis/external/marray/marray/utility* tblis/include/tblis/external/marray/marray/detail

echo "****************************"
echo "*** Installing Einsum IR ***"
echo "****************************"

git clone https://github.com/scalable-analyses/einsum_ir.git
cd einsum_ir
git log | head -n 25

wget https://github.com/catchorg/Catch2/releases/download/v2.13.10/catch.hpp

scons libtorch=../venv_pytorch/lib/python3.10/site-packages/torch blas=$(pwd)/../openblas tblis=$(pwd)/../tblis libxsmm=$(pwd)/../libxsmm -j8
mv build build_gcc

CXX=clang++ CC=clang scons libtorch=../venv_pytorch/lib/python3.10/site-packages/torch blas=$(pwd)/../openblas tblis=$(pwd)/../tblis libxsmm=$(pwd)/../libxsmm -j8
mv build build_llvm
cd ..