set -e

### Create and activate conda environment
conda env create -f environment.yml
conda activate gesture-recognition-rp2040
###

### Setup executorch toolchain
git clone --branch v0.3.0 https://github.com/pytorch/executorch.git
cd executorch

git submodule sync
git submodule update --init

./install_requirements.sh --pybind xnnpack
###

### Clean up
cd ..
rm -rf executorch
###
