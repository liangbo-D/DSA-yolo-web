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

    # 获取输出文件路径（图片或视频）
    output_path = str(results[0].save_dir / results[0].path.name)

    # 如果你还想在控制台打印路径
    print("✅ 模型输出文件：", output_path)

    return results, output_path


if __name__ == '__main__':
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'input.jpg'
    run_predict(input_path)
