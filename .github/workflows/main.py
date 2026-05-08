import anthropic
import os
import time

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

while True:
    try:
        print("Triggering YoniBot session...")
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            system="You are YoniBot, an autonomous quantitative trading intelligence system. Run your research loop fetching arXiv q-fin papers and extract tradeable anomalies. Then run your market loop screening live conditions. Send a summary to Telegram using the bot token and chat ID in your instructions.",
            messages=[{"role": "user", "content": "Begin. Run research loop then market loop. Send results to Telegram."}],
        )
        print("Done:", response.content[0].text[:200])
    except Exception as e:
        print(f"Error: {e}")
    print("Waiting 15 minutes...")
    time.sleep(900)
