import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'E:\myinclude\yolo\ultralytics\ultralytics-main\runs\train\yolov10s\weights\best.pt') 
    model.val(data=r'E:\myinclude\yolo\ultralytics\ultralytics-main\dataset\data.yaml',
              split='val', # 
              imgsz=640,
              batch=16,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )