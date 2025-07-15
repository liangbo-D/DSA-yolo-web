# app.py
from flask import Flask, request, jsonify
from predict import run_predict
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "✅ YOLO API 正在运行"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': '请上传图像文件'}), 400

    save_path = 'input.jpg'
    file.save(save_path)

    results = run_predict(save_path)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    return jsonify({'detections': boxes})

os.environ['YOLO_CONFIG_DIR'] = '/tmp'
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render 会自动设置 PORT 变量
    app.run(host='0.0.0.0', port=port)
