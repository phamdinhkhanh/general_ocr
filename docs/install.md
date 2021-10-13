# 1. Installation

## 1.1. Prerequisites

- Linux (Windows is not officially supported now)
- Python 3.7
- Pytorch >= 1.6
- torchvision 0.7.0
- CUDA 10.1

Our version OS and packages versions we used for bellow tests was:

- OS: Ubuntu 18.04
- CUDA: 10.1
- GCC (G++): 8.4.0 (CUDA 10.1 do not support for g++ version later 8)
- Pytorch: 1.6.0
- torchvision: 0.7.0

## 1.2. Installation introduction

1. Create conda virtual environment and afterward activate it.

```shell
conda create -n gen_ocr python=3.7 -y
conda activate gen_ocr
```

2. Install Pytorch and torchvision.
following the [official Pytorch instructions](https://pytorch.org/)

```shell
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

::: Note
Make sure that your CUDA compilation and CUDA runtime is matched version in order to avoid unexpected error. To check your CUDA runtime version in termial `nvcc --version`.
:::

3. Clone general_ocr repository

```shell
git clone https://github.com/phamdinhkhanh/general_ocr.git
cd general_ocr
```

4. Download `onnxruntime-linux` from ONNX Runtime [releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.8.1), extract it, expose `ONNXRUNTIME_DIR` and finally add the lib path to `LD_LIBRARY_PATH` as below:

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz

tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
cd onnxruntime-linux-x64-1.8.1
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

5. Install `onnxruntime`

```shell
pip install onnxruntime==1.8.1
```

6. Install build requirements packages.

```shell
cd .. # general_ocr root repository directory
pip install -r requirements.txt
GENERAL_OCR_WITH_OPS=1 GENERAL_OCR_WITH_ORT=1 python -m pip install -v -e .
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## 1.3. Full set-up Script

Full setup script is there:

```shell
# create and activate conda enviroment
conda create -n gen_ocr python=3.7 -y
conda activate gen_ocr

# clone general_ocr
git clone https://github.com/phamdinhkhanh/general_ocr.git
cd general_ocr

# download onnxruntime-linux and expose its lib to LD_LIBRARY_PATH
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz

tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
cd onnxruntime-linux-x64-1.8.1
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH

# install onnxruntime
pip install onnxruntime==1.8.1

# install requirements.txt
cd .. # general_ocr root repository directory
pip install -r requirements.txt
GENERAL_OCR_WITH_OPS=1 GENERAL_OCR_WITH_ORT=1 python -m pip install -v -e .
export PYTHONPATH=$(pwd):$PYTHONPATH
```

Run script below to test your installation is successful:

```shell
conda activate gen_ocr
python general_ocr/utils/ocr.py demo/demo_text_ocr.jpg --print-result --imshow --det PANet_IC15 --recog SEG
```