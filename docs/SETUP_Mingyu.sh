# git Fork
# pycharm VCS checkout
# pycharm Deployment

# INSTALL miniconda3
# conda create -n open-mmlab python=3.7 -y
# conda activate open-mmlab

# check that gcc version is not 4.8.5, use other versions (4.9.4, > 4 is better, 5.4.0) in /mnt/lustre/share/gcc/
# conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
# pip install cython
# pip install -r requirements.txt  # (include mmcv)

# rm -rf build & python setup.py develop

mkdir data
cd data
ln -s /mnt/lustre/share/DSK/datasets/mscoco2017 coco

wget https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet101_caffe-3ad79236.pth