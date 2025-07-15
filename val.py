import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('D:/python_program/yolov11/ultralytics-8.3.28/runs/detect/train20/weights/best.pt')
    model.val(data='D:/python_program/yolov11/ultralytics-8.3.28/data.yaml',
              imgsz=1024,
              batch=16,
              split='test',
              workers=14,
              device='0',
              conf=0.5,  # 置信度阈值
              save_json=True,
              save_txt=True,
              save_conf=True,
              )
