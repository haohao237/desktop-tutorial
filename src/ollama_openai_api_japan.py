from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

response = client.chat.completions.create(
  model="ELYZA",
  messages=[
  {"role": "system", "content": "あなたはAIのスペシャリストです。"},
  {"role": "user", "content": "AI資格試験に関する問題を以下の出力形式で10問作成してください。\n\n#出力形式\n-各問題は以下の形式で作成する：\n-穴埋め問題形式（例: ～を（　　）といいます）。\n-4つの選択肢（A）（B）（C）（D）を提示。\n-正解を明記。\n-簡単な解説を添える。"}
   ]
)
print(response.choices[0].message.content)