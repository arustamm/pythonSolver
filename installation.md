# SEP computation cluster
- username: `fpicetti`
- password: `sep123`
- login node: `oas.stanford.edu` (exposed to public IP)
- home directory is `/homes/sep/fpicetti` in the oas login node
- shared storage for scripts and software (like conda): `/net/server/sep/fpicetti`
- shared storage for datasets and results: `/net/brick6/data4/fpicetti`


### Install SEP Acoustic Isotropic Library to ISPL cluster

1. Install `cmake`
```bash
wget https://github.com/Kitware/CMake/releases/download/v3.14.5/cmake-3.14.5.tar.gz
tar -xzvf cmake-3.14.5.tar.gz
cd cmake-3.14.5
./bootstrap --prefix=/nas/home/fpicetti/cmake
make -j4
make install
cd .. && rm -rf cmake-3.14.5*
```

2. Install CUDA
```bash
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
bash cuda_10.1.243_418.87.00_linux.run --toolkit --installpath=/nas/home/fpicetti/cuda-10.1/
```

3. Create environment with dependencies
```bash
conda create -n sep python==3.7
conda activate sep
conda install pybind11 jupyter ipykernel ipython matplotlib -y
```

4. Install
```bash
sudo apt install -y libtbb-dev libboost-all-dev libboost-dev libgfortran flex
```

4. Install Library
```bash
git clone http://zapad.Stanford.EDU/barnier/acoustic_isotropic_operators.git
cd acoustic_isotropic_operators
git submodule update --init --recursive -- acoustic_iso_lib/external/ioLibs
git submodule update --init --recursive --remote acoustic_iso_lib/external/ioLibs
cd build
/nas/home/fpicetti/cmake/bin/cmake -DCMAKE_INSTALL_PREFIX=../sep_acoustic_iso_lib -DCMAKE_CUDA_COMPILER=/nas/home/fpicetti/cuda-10.1/bin/nvcc -DCMAKE_BUILD_TYPE=Debug ../acoustic_iso_lib/
make install -j8
```

5. Set up the paths
```bash
conda activate sep
cd miniconda3/envs/sep
mkdir ./etc/conda/activate.d
mkdir ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/envs_vars.sh
touch ./etc/conda/deactivate.d/envs_vars.sh
```
 - Using `nano ./etc/conda/activate.d/envs_vars.sh` write the following:
```bash
export PYTHONPATH='/nas/home/fpicetti/acoustic_isotropic_operators/sep_acoustic_iso_lib/lib/python/'
export DATAPATH='/nas/home/fpicetti/scratch'
export LD_LIBRARY_PATH='/nas/home/fpicetti/acoustic_isotropic_operators/sep_acoustic_iso_lib/lib'
PATH=/nas/home/fpicetti/acoustic_isotropic_operators/sep_acoustic_iso_lib/bin:$PATH
```
- Using `nano ./etc/conda/deactivate.d/envs_vars.sh` write the following:
```bash
unset DATAPATH
unset PYTHONPATH
unset LD_LIBRARY_PATH
PATH=$(echo :$PATH: | sed -e 's,:/nas/home/fpicetti/acoustic_isotropic_operators/sep_acoustic_iso_lib/lib:,:,g' -e 's/^://' -e 's/:$//')
```
