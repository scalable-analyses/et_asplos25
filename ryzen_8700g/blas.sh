# echo "**********************************"
# echo "*** Installing BLAS Playground ***"
# echo "**********************************"
# cd blas
# CXX=clang++ scons libtorch=../venv_pytorch/lib/python3.10/site-packages/torch blas=$(pwd)/../openblas tblis=$(pwd)/../tblis libxsmm=$(pwd)/../libxsmm