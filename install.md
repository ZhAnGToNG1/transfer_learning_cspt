## Our work environment：


- Linux 18.04
- Python 3.7+
- PyTorch 1.7.1
- CUDA 11.0

## Installation：

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n dspt python=3.7 -y
    conda activate dspt
    ```
    
2. Install other dependencies,
    ```shell
    conda install -c pytorch pytorch==1.7.1 torchvision==0.8.2
    pip install timm=0.3.2
    pip install tensorboard
    ```
    
3. For detection, you can refer mmdetection,
    ```shell
    pip install openmim
    mim install mmcv-full
    cd finetune/detection
    pip install -r requirements/build.txt
    pip install -v -e .
    ```
4. For segmentation, you can refer beit:
    ```shell
    pip install mmsegmentation
    pip install scipy

    Note: Install apex for mixed-precision training
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    ```
