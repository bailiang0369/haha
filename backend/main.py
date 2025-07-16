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

def get_klines_data():
    data = [
        [1622505600000, '49000', '49100', '48900', '49050', '1000', 1622509199999, '49000000', 100, '500', '24500000', '0'],
        [1622509200000, '49050', '49200', '49000', '49150', '1200', 1622512799999, '59000000', 120, '600', '29500000', '0'],
        [1622512800000, '49150', '49300', '49100', '49250', '1300', 1622516399999, '64000000', 130, '650', '31850000', '0'],
        [1622516400000, '49250', '49400', '49200', '49350', '1400', 1622519999999, '69000000', 140, '700', '34300000', '0'],
        [1622520000000, '49350', '49500', '49300', '49450', '1500', 1622523599999, '74000000', 150, '750', '36750000', '0'],
        # Add more data for MACD calculation
        [1622523600000, '49450', '49600', '49400', '49550', '1600', 1622527199999, '79000000', 160, '800', '39200000', '0'],
        [1622527200000, '49550', '49700', '49500', '49650', '1700', 1622530799999, '84000000', 170, '850', '41650000', '0'],
        [1622530800000, '49650', '49800', '49600', '49750', '1800', 1622534399999, '89000000', 180, '900', '44100000', '0'],
        [1622534400000, '49750', '49900', '49700', '49850', '1900', 1622537999999, '94000000', 190, '950', '46550000', '0'],
        [1622538000000, '49850', '50000', '49800', '49950', '2000', 1622541599999, '99000000', 200, '1000', '49000000', '0'],
        [1622541600000, '49950', '50100', '49900', '50050', '2100', 1622545199999, '104000000', 210, '1050', '51450000', '0'],
        [1622545200000, '50050', '50200', '50000', '50150', '2200', 1622548799999, '109000000', 220, '1100', '53900000', '0'],
        [1622548800000, '50150', '50300', '50100', '50250', '2300', 1622552399999, '114000000', 230, '1150', '56350000', '0'],
        [1622552400000, '50250', '50400', '50200', '50350', '2400', 1622555999999, '119000000', 240, '1200', '58800000', '0'],
        [1622556000000, '50350', '50500', '50300', '50450', '2500', 1622559599999, '124000000', 250, '1250', '61250000', '0'],
        [1622559600000, '50450', '50600', '50400', '50550', '2600', 1622563199999, '129000000', 260, '1300', '63700000', '0'],
    ]
    df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

    df['time'] = df['open_time'] / 1000
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df

@app.get("/api/klines")
def get_klines(symbol: str = 'BTCUSDT', interval: str = '1h'):
    df = get_klines_data()
    return json.loads(df[['time', 'open', 'high', 'low', 'close', 'volume']].to_json(orient='records'))

@app.get("/api/macd")
def get_macd(symbol: str = 'BTCUSDT', interval: str = '1h', fastperiod=12, slowperiod=26, signalperiod=9):
    df = get_klines_data()

    exp1 = df['close'].ewm(span=fastperiod, adjust=False).mean()
    exp2 = df['close'].ewm(span=slowperiod, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    histogram = macd - signal

    macd_data = pd.DataFrame({
        'time': df['time'],
        'macd': macd,
        'signal': signal,
        'histogram': histogram
    }).dropna()

    return json.loads(macd_data.to_json(orient='records'))
