########################################################################################
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh -O $HOME/conda.sh

bash $HOME/conda.sh -bf -p $HOME/conda/

CONDA_DIR=$HOME/conda/

echo "export "CONDA_DIR=${CONDA_DIR}"" >> $HOME/.bashrc

source $CONDA_DIR/bin/activate
########################################################################################

conda create -n gate python=3.8 -y
conda activate gate


conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-nightly -y
conda install opencv -y
conda install h5py -y

#conda install -c conda-forge git-lfs -y
# optional conda install starship tmux -y
conda install gh --channel conda-forge -y
#apt install htop nvtop -y
#apt-get install ffmpeg libsm6 libxext6  -y

# optional conda install bat micro -y
########################################################################################
echo "export CODE_DIR=$HOME/target_codebase" >> $HOME/.bashrc
echo "export MOUNT_DIR=/mnt/disk/tali/" >> $HOME/.bashrc
#echo "export MOUNT_DIR=/mnt/scratch_ssd/antreas" >> $HOME/.bashrc
echo "export EXPERIMENTS_DIR=$MOUNT_DIR/experiments/" >> $HOME/.bashrc
echo "export DATASET_DIR=$MOUNT_DIR/dataset/" >> $HOME/.bashrc
echo "export TOKENIZERS_PARALLELISM=false" >> $HOME/.bashrc
echo "export FFREPORT=file=ffreport.log:level=32" >> $HOME/.bashrc
echo "export OPENCV_LOG_LEVEL=SILENT" >> $HOME/.bashrc
echo "export TMPDIR=$MOUNT_DIR/tmp" >> $HOME/.bashrc

echo "source $CONDA_DIR/bin/activate" >> $HOME/.bashrc
echo "conda activate tali" >> $HOME/.bashrc

source $HOME/.bashrc
########################################################################################
cd $HOME
git clone https://github.com/AntreasAntoniou/TALI-lightning-hydra.git $CODE_DIR
cd $CODE_DIR

pip install -r $CODE_DIR/requirements.txt
pip install -e $CODE_DIR

#cd $HOME
#git clone https://huggingface.co/openai/clip-vit-base-patch32

########################################################################################

 # ~/.config/starship.toml

