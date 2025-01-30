echo "**********************************"
echo "*** Installing System Packages ***"
echo "**********************************"

sudo dnf install -y vim htop gfortran clang wget git cmake environment-modules g++-13 python3.10 scons tmux

echo "**************************"
echo "*** Installing oneDNN ***"
echo "**************************"

git clone https://github.com/oneapi-src/oneDNN.git
cd oneDNN
git checkout v3.6.1
mkdir build
cd build
cmake ..
make -j96