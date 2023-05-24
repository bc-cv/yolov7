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
- If no `curl` on your system, download the model weight with [[this link]](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) and move it under this folder 

- The object classes that can be detected are: 
```
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush
```
