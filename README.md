# 🛸 Soul Flight Recorder (SFR)

> **「AIと会話するだけで、あなたの内側にある『本当にやりたいこと』が見えてくる」**
> 就活生や、これから何をしようか迷っている学生のための対話型AI自己分析ツール。

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://soul-flight-recorder-chqfobdjq8lz2rydqnbbe7.streamlit.app/)

## 🌐 アプリを試す
インストール不要！以下のリンクからすぐにブラウザで利用できます。

👉 **[🚀 Soul Flight Recorder を起動する](https://soul-flight-recorder-chqfobdjq8lz2rydqnbbe7.streamlit.app/)**

---

## 💡 このツールについて

**「自分は本当に何がしたいんだろう？」**

宇宙イベントの運営、ロボコン、海外ハッカソン——多くの挑戦を重ねるうちに、活動の数が増えるほど、逆に自分の軸が見えにくくなる。そんな原体験から生まれたのが Soul Flight Recorder です。

以前は「過去の日記やメモ」を自分で用意して読み込ませる必要がありましたが、**最新版では一切の事前準備が不要**になりました。
AIからの簡単な質問にチャット形式で答えていくだけで、あなたの回答の裏側にある「ワクワクのパターン」や「潜在的な強み」をAIが自動でプロファイリングし、5年後のビジョンや次の一歩を提案します。

---

## ✨ 主な機能と特徴

| 機能 | 説明 |
|------|------|
| 🗣️ **対話型インタビュー** | 白紙から考える必要はありません。AIの質問に答えるだけで深層心理を引き出します。 |
| 🧠 **バックグラウンド解析** | 会話の裏側で、LLMがあなたの回答を自動的に構造化し、パーソナルプロファイルを構築します。 |
| 🌱 **Ecoモード（トークン節約）** | 長時間の対話でもAPI制限に引っかからないよう、履歴をスマートに要約するモードを搭載。 |
| 🔐 **BYOK（APIキー持ち込み）** | ユーザー自身のGroq APIキーを使用するセキュアな設計。開発者にデータやキーが漏れることはありません。 |
| 🌍 **多言語対応** | 日本語と英語のUI・対話モードをシームレスに切り替え可能。 |

---

## 📖 使い方（Webアプリ版）

誰でも簡単に、今すぐ自己分析を始められます。

1. **[アプリを開く](https://soul-flight-recorder-chqfobdjq8lz2rydqnbbe7.streamlit.app/)**
2. **APIキーを用意する**
   - [Groq Console](https://console.groq.com/keys) にアクセスし、無料のAPIキーを発行します。
3. **設定を入力**
   - アプリのサイドバーにある「🔑 Groq API Key」欄に取得したキーを貼り付けます。
   - お好みで「Ecoモード（節約）」や言語を選択してください。
4. **AIと会話スタート！**
   - AIから最初の質問が飛んできます。直感で答えていきましょう。
   - 会話が進むにつれて、あなた専用の「未来予想」や「強み」が明確になっていきます！

---

## 🛠 技術スタック

- **LLM**: [Groq API](https://groq.com/) (openai/gpt-oss-120b を利用し爆速レスポンスを実現)
- **UI/フロントエンド**: Streamlit
- **言語**: Python 3.11
- **アーキテクチャ**: LLMを活用した動的プロンプト生成 ＆ コンテキスト要約処理

---

## 💻 ローカルでの開発・実行方法

開発者向けの手順です。手元でコードを動かしたい場合はこちら。

```bash
# 1. リポジトリをクローン
git clone [https://github.com/0822-inter/soul-flight-recorder.git](https://github.com/0822-inter/soul-flight-recorder.git)
cd soul-flight-recorder

# 2. 仮想環境の作成と依存関係のインストール
python -m venv venv
source venv/bin/activate  # Windowsの場合は venv\Scripts\activate
pip install -r requirements.txt

# 3. 起動
streamlit run app.py
