# AI Feedback System

このプロジェクトは、AI検定の問題に対するユーザーの解答結果を基に、機械学習を使用して動的にフィードバックを生成するシステムです。GPT系のモデルを使用せず、正誤判定や解答時間を特徴量として学習し、個別のフィードバックを作成します。

## プロジェクトの構成

- `data/`: 学習データおよび出力されたフィードバックを保存するディレクトリ
- `models/`: 学習済みモデルを保存するディレクトリ
- `src/`: ソースコードのディレクトリ
  - `train_model.py`: モデルの学習スクリプト
  - `analyze_feedback.py`: データ解析およびフィードバック生成
  - `feedback_generator.py`: 実際にフィードバックを生成するスクリプト
- `Dockerfile`: Docker環境設定
- `requirements.txt`: 必要なPythonライブラリ

## セットアップ手順

1. リポジトリをクローンします。
2. `requirements.txt` を使って必要なライブラリをインストールします。

```bash
pip install -r requirements.txt
