import argparse
import numpy as np
import cv2
import torch
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

class TrafficDetector:
    def __init__(self):
        self.load_default_conf()

    def set_conf(self, attr_name, attr_value):
        setattr(self.opt, attr_name, attr_value)

    def load_default_conf(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')                
        parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
        self.opt = parser.parse_args(['--nosave'])

    def setup_model(self):
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        weights, imgsz, trace = self.opt.weights, self.opt.img_size, not self.opt.no_trace
        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
        if trace:
            self.model = TracedModel(self.model, self.device, self.opt.img_size)

        if self.half:
            self.model.half()  # to FP16

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
        self.cls_select = np.array([2,3,5,7]) # car,motocycle,bus,truck
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def detect(self, img0, output_ratio=1):
        # img0: cv2.imread
        img0 = img0.copy()
        # img: Padded resize
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)                
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres,\
                                   classes=self.opt.classes, agnostic=self.opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image            
            if len(det):
                # Rescale boxes from img_size to img size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if int(cls) in self.cls_select:
                      label = f'{self.names[int(cls)]} {conf:.2f}'
                      plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=3)
        if output_ratio != 1:
            img0 = img0[::output_ratio,::output_ratio]
        return img0

