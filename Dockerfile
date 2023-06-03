# ベースイメージを指定
FROM python:3.10

# 作業ディレクトリを設定
WORKDIR /app

# 依存パッケージをインストール
COPY app/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# アプリケーションのコードを追加
COPY app/ .

# デフォルトのポートを公開
EXPOSE 5000

# アプリケーションを起動
CMD ["python", "app.py"]

