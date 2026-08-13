from google import genai

client = genai.Client()

for model in client.models.list():
    # Filter model yang mendukung batch content generation
    if "batchGenerateContent" in (model.supported_actions or []):
        print(f"Supported Batch Model: {model.name}")
