from flask import Flask, request, send_file, jsonify
from predict import run_predict
import os
from flask_cors import CORS
import shutil
import uuid

app = Flask(__name__)
CORS(app)

# 避免 Ultralytics 写入只读路径
os.environ['YOLO_CONFIG_DIR'] = '/tmp'

@app.route('/')
def home():
    return "✅ YOLO API 正在运行"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': '请上传图像或视频文件'}), 400

    # 保存上传文件
    file_ext = os.path.splitext(file.filename)[1].lower()
    unique_id = str(uuid.uuid4())
    filename = f"input_{unique_id}{file_ext}"
    save_path = os.path.join('/tmp', filename)
    file.save(save_path)

    # 运行 YOLO 推理
    results = run_predict(save_path)

    # 获取保存的图像/视频路径（默认保存在 runs/predict/exp/）
    output_path = results[0].save_dir / results[0].path.name
    output_path = str(output_path)

    # 设置合适的 MIME 类型
    if file_ext in ['.mp4', '.avi', '.mov']:
        mimetype = 'video/mp4'
    else:
        mimetype = 'image/jpeg'

    return send_file(output_path, mimetype=mimetype)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
