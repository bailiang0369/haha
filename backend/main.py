from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json

app = FastAPI()

# a very permissive CORS policy, this is for development only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/klines")
def get_klines(symbol: str = 'BTCUSDT', interval: str = '1h'):
    # In a real application, you would fetch this data from Binance
    # For now, we'll use some sample data
    data = [
        [1622505600000, '49000', '49100', '48900', '49050', '1000', 1622509199999, '49000000', 100, '500', '24500000', '0'],
        [1622509200000, '49050', '49200', '49000', '49150', '1200', 1622512799999, '59000000', 120, '600', '29500000', '0'],
        [1622512800000, '49150', '49300', '49100', '49250', '1300', 1622516399999, '64000000', 130, '650', '31850000', '0'],
        [1622516400000, '49250', '49400', '49200', '49350', '1400', 1622519999999, '69000000', 140, '700', '34300000', '0'],
        [1622520000000, '49350', '49500', '49300', '49450', '1500', 1622523599999, '74000000', 150, '750', '36750000', '0']
    ]
    df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

    # Convert to format that lightweight-charts can understand
    df['time'] = df['open_time'] / 1000
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    return json.loads(df[['time', 'open', 'high', 'low', 'close', 'volume']].to_json(orient='records'))
