# YOLOv7 for ISU 2023 AI Workshop

[Official YOLOv7](https://github.com/WongKinYiu/yolov7)

## Installation

- Install `conda`: [[link to miniconda]](https://docs.conda.io/en/latest/miniconda.html)
- On the terminal:
``` shell
conda create -n yolov7
source activate yolov7

# pip install required packages
conda install pip
pip install -r requirements.txt
```

## Demo

- Launch jupyterlab and open demo.ipynb
``` shell
jupyter lab 
```
- If no `wget` on your system, download the model weight with [[this link]](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) and move it under this folder 
