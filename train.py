import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('runs/train/exp/weights/last.pt')
    model = YOLO(r'C:\Users\clown\Desktop\GTAIL_8_24042025-1141902\新建文件夹\Code\ultralytics-main\ultralytics\cfg\models\v10\yolov10n-STFPN-TSAH-CARAFE.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=r'E:\myinclude\yolo\ultralytics\ultralytics-main\dataset\data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=0,
                workers=8, 
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                #resume=True, 
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )