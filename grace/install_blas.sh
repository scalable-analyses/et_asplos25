eval "$(conda shell.bash hook)"

echo "*******************************"
echo "*** Installing Dependencies ***"
echo "*******************************"
git clone https://github.com/libxsmm/libxsmm.git
cd libxsmm
git log | head -n 25
make BLAS=0 -j
cd ..

python -m venv venv_pytorch
source venv_pytorch/bin/activate
pip3 install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__config__.show()); print(torch.__config__.parallel_info());"
ln -s $(pwd)/venv_pytorch/lib64/python3.9/site-packages/torch.libs/* venv_pytorch/lib64/python3.9/site-packages/torch/lib/

echo "**********************************"
echo "*** Installing BLAS Playground ***"
echo "**********************************"
cd blas
CXX=clang++ scons libtorch=../venv_pytorch/lib/python3.9/site-packages/torch blas=yes tblis=no libxsmm=$(pwd)/../libxsmm
cd ..