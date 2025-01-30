eval "$(conda shell.bash hook)"

echo "*****************************"
echo "*** Installing opt_einsum ***"
echo "*****************************"

conda create -p ./conda_oe python=3.10 -y
conda activate ./conda_oe

pip install opt-einsum==3.4.0
python -c "import opt_einsum; print( opt_einsum.__version__ );"
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__config__.show()); print(torch.__config__.parallel_info());"

git clone https://github.com/scalable-analyses/einsum_ir.git einsum_ir
cd einsum_ir
git log | head -n 25
cd ..