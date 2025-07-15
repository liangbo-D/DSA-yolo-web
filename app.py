from flask import Flask, request, Response, jsonify, send_file
from predict import run_predict
import os
import uuid
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

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

    # 运行推理并获取输出文件路径
    results, output_path = run_predict(input_path)
    pred = results[0]

    # 提取检测框数据
    boxes = pred.boxes.xyxy.cpu().tolist() if pred.boxes else []
    confs = pred.boxes.conf.cpu().tolist() if pred.boxes else []
    clss = pred.boxes.cls.cpu().tolist() if pred.boxes else []

    detections = [
        {"box": box, "confidence": conf, "class": int(cls)}
        for box, conf, cls in zip(boxes, confs, clss)
    ]

    # 返回视频或图像文件
    mimetype = 'video/mp4' if ext in ['.mp4', '.avi', '.mov'] else 'image/jpeg'

    with open(output_path, 'rb') as f:
        data = f.read()

    resp = Response(data, mimetype=mimetype)
    resp.headers['X-Detections'] = json.dumps(detections)
    return resp

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
