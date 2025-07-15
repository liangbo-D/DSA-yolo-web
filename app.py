from flask import Flask, request, send_file, jsonify
from ultralytics import YOLO
import os
import uuid
import shutil
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 加载模型
model = YOLO("best.pt")

@app.route('/')
def home():
    return "✅ YOLO Flask API 已启动"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "请上传文件"}), 400

    # 保存上传文件
    ext = os.path.splitext(file.filename)[1].lower()
    uid = str(uuid.uuid4())[:8]
    input_path = f"input_{uid}{ext}"
    file.save(input_path)

    # 运行推理
    results = model.predict(source=input_path, imgsz=640, save=True)
    pred = results[0]

    # 获取预测图像或视频文件路径
    output_path = pred.save_dir / pred.path.name
    output_path = str(output_path)

    # 提取检测框信息
    boxes = pred.boxes.xyxy.cpu().tolist() if pred.boxes else []
    confs = pred.boxes.conf.cpu().tolist() if pred.boxes else []
    clss = pred.boxes.cls.cpu().tolist() if pred.boxes else []

    detections = [
        {"box": box, "confidence": conf, "class": int(cls)}
        for box, conf, cls in zip(boxes, confs, clss)
    ]

    # 设置 MIME 类型
    if ext in ['.mp4', '.avi', '.mov']:
        mimetype = 'video/mp4'
    else:
        mimetype = 'image/jpeg'

    # 将 detections 作为 JSON 编码在自定义响应头中返回
    from flask import Response
    f = open(output_path, 'rb')
    data = f.read()
    f.close()

    from json import dumps
    resp = Response(data, mimetype=mimetype)
    resp.headers['X-Detections'] = dumps(detections)
    return resp

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
