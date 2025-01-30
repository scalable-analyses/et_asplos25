eval "$(conda shell.bash hook)"

echo "*******************************"
echo "*** Installing Dependencies ***"
echo "*******************************"
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

python -m venv venv_pytorch
source venv_pytorch/bin/activate
pip3 install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__config__.show()); print(torch.__config__.parallel_info());"

echo "**********************************"
echo "*** Installing BLAS Playground ***"
echo "**********************************"
cd blas
CXX=g++ scons libtorch=../venv_pytorch/lib/python3.10/site-packages/torch blas=$(pwd)/../openblas tblis=no libxsmm=$(pwd)/../libxsmm
cd ..