import io
import os
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import base64
from src.model import Model


app = Flask(__name__, static_folder='static')

# 静的ファイルの提供を有効にする
app.use_static_for_external=True

UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
MAX_IMAGES = 5

model = Model()
device = torch.device('cpu')
torch.load('./alex_net.pth', map_location=device)
model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transform_image(file):
    img = Image.open(file).convert('RGB')
    img = img.resize((256, 256))  # 必要なサイズに変更してください
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.44,), (0.22,))])  # 学習時と同様に正規化
    
    return transform(img).unsqueeze(0)


def get_prediction(file):
    tensor = transform_image(file=file)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)  # スコアが最大の数字を予測値として取得

    return y_hat.item()


@app.route('/upload', methods=['POST'])
def upload():
    # アップロードされたファイルを取得します
    file = request.files['file']
    if not file:
        return render_template('index.html', base64_data_home=None)
    elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            _img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(_img_path)
    
    prediction = get_prediction(file=file)  # 数字の予測
    
    # 予測結果を表示します
    if prediction == 0:
        result = 'No Goldfish'
    else:
        result = 'Goldfish'
    # _img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    # file.save(_img_path)
    with open(_img_path, 'rb') as f:
        image_data = f.read()
        base64_data = base64.b64encode(image_data).decode('utf-8')

    # アップロードされた画像の数をカウント
    image_count = len(os.listdir(UPLOAD_FOLDER))
    
    # 5枚以上の画像がある場合は古い画像を削除
    if image_count > MAX_IMAGES:
        images = sorted(os.listdir(UPLOAD_FOLDER))
        for image in images:
            os.remove(os.path.join(UPLOAD_FOLDER, image))
    
    # テンプレートに画像のパスを渡す
    return render_template('result.html', result=result, base64_data=base64_data, base64_data_home=None)
    
@app.route('/')
def index():
    with open('static/home/00008.jpg', 'rb') as f:
        image_data = f.read()
        base64_data = base64.b64encode(image_data).decode('utf-8')
    return render_template('index.html', base64_data_home=base64_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)