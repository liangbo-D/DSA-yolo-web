# app.py
from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = YOLO("best.pt")  # 载入模型文件

@app.route('/')
def home():
    return "✅ YOLO Flask API 已启动"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if file is None:
        return jsonify({"error": "请上传文件"}), 400

    temp_path = 'temp_input'
    os.makedirs(temp_path, exist_ok=True)
    input_path = os.path.join(temp_path, file.filename)
    file.save(input_path)

    results = model.predict(source=input_path, imgsz=640)
    detections = results[0].boxes.xyxy.cpu().tolist()
    return jsonify({"detections": detections})



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
