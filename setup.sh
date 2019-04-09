# gpu==nvidia 1050ti driver_version==418.56
conda create -n chinese-ocr python=3.6 pip scipy numpy pillow jupyter  ##运用conda 创建python环境 建议先行添加tuna源
conda activate chinese-ocr
conda install cudatoolkit=9.2
conda install cudnn=7.3.1
pip install easydict -i https://pypi.tuna.tsinghua.edu.cn/simple/ ##选择国内源，速度更快
pip install Cython -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install -U pillow -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install  h5py lmdb mahotas -i https://pypi.tuna.tsinghua.edu.cn/simple/
#conda install pytorch=1.0.1 torchvision -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/   Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
pip install torch torchvision
conda install tensorflow=1.10 tensorflow-gpu=1.10
pip install keras==2.0.8  -i https://pypi.tuna.tsinghua.edu.cn/simple/ ##解决cuda报错相关问题
conda install py-opencv=3.4.2
cd ./ctpn/lib/utils
sh make.sh


