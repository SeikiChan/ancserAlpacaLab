from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("APCA_API_KEY_ID")
secret_key = os.getenv("APCA_API_SECRET_KEY")

client = TradingClient(api_key, secret_key, paper=True)

print("Attributes of TradingClient:")
for attr in dir(client):
    if not attr.startswith("_"):
        print(attr)
