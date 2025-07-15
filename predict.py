from ultralytics import YOLO
import sys
import os

def run_predict(input_path='input.jpg'):
    model = YOLO('best.pt')  # 模型路径相对或远程

    results = model.predict(
        source=input_path,  # 图像或视频路径
        save=True,          # 保存带框结果图像
        save_txt=True,      # 保存 txt 检测框
        conf=0.5,           # 置信度阈值
        imgsz=1024,         # 输入尺寸
        device='cpu'        # Render 上默认用 CPU
    )

    # 获取输出文件路径
    pred = results[0]
    output_path = str(pred.save_dir / pred.path.name)
    return results, output_path


if __name__ == '__main__':
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'input.jpg'
    results, output_path = run_predict(input_path)
    print(f"预测输出文件路径: {output_path}")
