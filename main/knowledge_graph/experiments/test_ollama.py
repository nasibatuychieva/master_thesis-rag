import ollama

resp = ollama.chat(
    model='qwen3:4b',    # exakt wie in `ollama list`
    messages=[{"role": "user", "content": "Erkläre mir RAG kurz."}],
)

print(resp["message"]["content"])
