import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics import RTDETR

if __name__ == '__main__':
    model = YOLO(r'D:\yolo11_wdq\ultralytics-8.3.90\change_yamls\SAAC+SER+MFFPN.yaml')
    model.train(data=r'D:\yolo11_wdq\ultralytics-8.3.90\dataset_2\mydata.yaml',
                cache=False,
                imgsz=640,
                epochs=400,
                batch=16,
                workers=8,
                line_width=1,
                patience=30,
                warmup_epochs=3,
                project='runs_final/seed=0/ours-s',
                # project='runs_try/try',
                # seed=42,
                # seed=2023,
                # seed=3407,
                # amp=False
                )