from openai import OpenAI

client = OpenAI(
    base_url="http://10.5.32.17:11434/v1",
    api_key="ollama"
)

response = client.chat.completions.create(
    model="llama3.1:latest",
    messages=[{"role": "user", "content": "Say hello"}],
)

print(response.choices[0].message.content)
