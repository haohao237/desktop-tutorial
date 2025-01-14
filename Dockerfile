FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# NVIDIA Container Runtimeの設定（推奨形式に修正）
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# 必要なPythonパッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのソースコードをコピー
COPY . .

# デフォルトのコマンドを設定
CMD ["bash"]
