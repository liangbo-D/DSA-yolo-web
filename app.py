from flask import Flask, request, send_file, jsonify
from predict import run_predict
import os
from flask_cors import CORS
import shutil
import uuid
import glob
import time

app = Flask(__name__)
CORS(app)

# 设置配置目录
os.environ['YOLO_CONFIG_DIR'] = '/tmp'

UPLOAD_DIR = '/tmp'
EXPIRE_SECONDS = 600  # 10分钟后删除文件


def cleanup_old_files():
    now = time.time()
    for pattern in ['input_*.jpg', 'input_*.mp4', 'runs/predict/exp*/**']:
        for path in glob.glob(os.path.join(UPLOAD_DIR, pattern), recursive=True):
            if os.path.isfile(path) and now - os.path.getmtime(path) > EXPIRE_SECONDS:
                try:
                    os.remove(path)
                except Exception:
                    pass
            elif os.path.isdir(path) and now - os.path.getmtime(path) > EXPIRE_SECONDS:
                try:
                    shutil.rmtree(path)
                except Exception:
                    pass


@app.route('/')
def home():
    return "✅ YOLO API 正在运行"


@app.route('/predict', methods=['POST'])
def predict():
    cleanup_old_files()

    file = request.files.get('file')
    if not file:
        return jsonify({'error': '请上传图像或视频文件'}), 400

    # 保存上传文件
    file_ext = os.path.splitext(file.filename)[1].lower()
    unique_id = str(uuid.uuid4())
    filename = f"input_{unique_id}{file_ext}"
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    # 运行推理
    results = run_predict(save_path)
    pred = results[0]

    # 获取 YOLO 默认输出图像或视频路径
    output_path = str(pred.save_dir / pred.path.name)

    # 提取预测框信息（坐标、类别、置信度）
    detection_data = []
    if pred.boxes:
        boxes = pred.boxes.xyxy.cpu().tolist()
        confs = pred.boxes.conf.cpu().tolist()
        clss = pred.boxes.cls.cpu().tolist()
        for box, conf, cls_id in zip(boxes, confs, clss):
            detection_data.append({
                'box': box,
                'confidence': conf,
                'class': int(cls_id)
            })

    # 响应图像/视频和 JSON
    file_type = 'video/mp4' if file_ext in ['.mp4', '.avi', '.mov'] else 'image/jpeg'
    return send_file(output_path, mimetype=file_type,
                     as_attachment=False,
                     download_name=os.path.basename(output_path),
                     headers={
                         'X-Detections': jsonify(detection_data).get_data(as_text=True)
                     })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
