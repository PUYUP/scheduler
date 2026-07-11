from google import genai


def generate_gemini_webhook():
    client = genai.Client()

    webhook = client.webhooks.create(
        name="Summarize Paper Batch Webhook",
        subscribed_events=["batch.succeeded", "batch.failed", "batch.expired"],
        uri="https://tunnel.atlanize.com/gemini-batch-webhook",
    )

    webhook_secret = webhook.new_signing_secret
    print(f"Created webhook: {webhook.name}, {webhook.id}")
    print(f"Webhook Secret Key: {webhook_secret}")


if __name__ == "__main__":
    generate_gemini_webhook()
