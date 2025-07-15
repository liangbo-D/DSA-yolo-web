import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('D:/python_program/yolov11/ultralytics-8.3.28/runs/detect/train9/weights/best.pt')
    model.predict(source='D:/detectimages',
                  imgsz=1024,
                  conf=0.3,  # 置信度阈值
                  device='0',
                  save=True,
                  )
