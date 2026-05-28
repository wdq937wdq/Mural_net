import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO(r'D:\yolo11_wdq\ultralytics-8.3.90\runs_final\seed=42\SER+MFFPN\train\weights\best.pt') # select your model.pt path
    model = YOLO(r'D:\yolo11_wdq\ultralytics-8.3.90\runs_posun\seed=0\ours-s\train2\weights\best.pt') # select your model.pt path
    model.predict(source=r"D:\yolo11_wdq\ultralytics-8.3.90\dataset_posun\DH\test\images\14.jpg",
                  imgsz=512,
                  # project=r'D:\云大\小论文\可视化结果\对比试验检测图\yolov8n',
                  project=r'D:\云大\小论文\可视化结果\posunjiance',
                  # name='MANet_GCConv+CGAFusion',
                  save=True,
                  conf=0.2,
                  # iou=0.1,
                  # agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                  retina_masks=True,

                  )