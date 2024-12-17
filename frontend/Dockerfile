# ベースイメージ
FROM node:18

# 作業ディレクトリを設定
WORKDIR /app

# パッケージファイルをコピーして依存関係をインストール
COPY package.json package-lock.json ./
RUN npm install

# ソースコードをコピー
COPY . .

# デフォルトポートを公開
EXPOSE 5173

# サーバーを起動
CMD ["npm", "run", "dev"]
