echo "****************************"
echo "*** Printing System Info ***"
echo "****************************"

date
uname -a
hostnamectl
g++ --version
clang++ --version
cat /usr/include/nvpl_blas_version.h
lscpu
lsmem