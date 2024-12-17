FROM python:3.11.10
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 必要なライブラリのインストール
RUN pip install --upgrade pip
RUN pip install "transformers[torch]" pandas scikit-learn
# 作業ディレクトリの設定
WORKDIR /app

# 必要なファイルをコピー
COPY requirements.txt .
COPY feedback.py /app/
COPY ../results /app/results

# 必要なパッケージをインストール
RUN pip install -r requirements.txt

# スクリプトをコピー
COPY . .

# Flaskサーバーを起動
CMD ["python", "feedback.py"]
