#%%
import pytest 
import websocket
import pandas as pd
import json

global df, in_position, use_features, model
df = pd.DataFrame()
in_position=False
model=None
use_features=[]
# TODO: Ladda in model och use_features från filer


def momentum_strategy(df):
    # Beräkna sma_5 och sma_20
    df['sma_5'] = df['pris'].rolling(5).mean()
    df['sma_20'] = df['pris'].rolling(20).mean()

    # Välj den senaste priset och sma-värdena
    current_price = df['pris'][-1]
    sma_5 = df['sma_5'][-1]
    sma_20 = df['sma_20'][-1]

    # Om senaste priset är högre än sma_5 och sma_20, returnera "buy"
    if current_price > sma_5 and current_price > sma_20:
        return "buy"
    # Om senaste priset är lägre än sma_5 och sma_20, returnera "sell"
    elif current_price < sma_5 and current_price < sma_20:
        return "sell"
    # Annars, returnera "hold"
    else:
        return "hold"


def fibonacci_strategy(df):
    # Beräkna 38.2%, 50% och 61.8% Fibonacci-nivåer för den senaste trenden
    # Högsta priset under de senaste 21 dagarna
    high = df['pris'].rolling(window=21).max()
    # Lägsta priset under de senaste 21 dagarna
    low = df['pris'].rolling(window=21).min()
    diff = high - low
    df['fib38'] = high - (0.382 * diff)
    df['fib50'] = high - (0.5 * diff)
    df['fib62'] = high - (0.618 * diff)

    # Köp när priset bryter igenom den 38.2% Fibonacci-nivån från botten till toppen
    if df['pris'].iloc[-1] > df['fib38'].iloc[-1] and df['pris'].iloc[-2] < df['fib38'].iloc[-2]:
        return 'buy'

    # Sälj när priset bryter igenom den 61.8% Fibonacci-nivån från toppen till botten
    elif df['pris'].iloc[-1] < df['fib62'].iloc[-1] and df['pris'].iloc[-2] > df['fib62'].iloc[-2]:
        return 'sell'

    # Behåll positionen annars
    else:
        return 'hold'


def get_b_strategy(row):
    i = row.name
    if i == pd.to_datetime('2022-01-24 00:00:00'):
        print('i = ',i, 'row = ', row,'\n', df['pris'].shift(1).loc[i], df['lower_band'].shift(1).loc[i], df['pris'].shift(0).loc[i], df['lower_band'].shift(0).loc[i])
    if df['pris'].shift(1).loc[i] < df['upper_band'].shift(1).loc[i] and df['pris'].shift(0).loc[i] > df['upper_band'].shift(0).loc[i]:
        return 'sell'
    elif df['pris'].shift(1).loc[i] > df['lower_band'].shift(1).loc[i] and df['pris'].shift(0).loc[i] < df['lower_band'].shift(0).loc[i]:
        return 'buy'
    else:
        return 'hold'

    
def bollinger_strategy(df):
    # Beräkna 20-dagars rullande medelvärde och standardavvikelse
    df['boll_sma'] = df['pris'].rolling(window=20).mean()
    df['boll_std'] = df['pris'].rolling(window=20).std()

    # Beräkna övre och undre Bollinger-bands
    df['upper_band'] = df['boll_sma'] + (2 * df['boll_std'])
    df['lower_band'] = df['boll_sma'] - (2 * df['boll_std'])

    # Avgör om vi ska köpa, sälja eller behålla
    # shift(-1) gör att vi jämför med nästa värde
    df['b_strategy'] = df.apply(get_b_strategy, axis=1)
   
    if len(df) < 2:
        return 'hold'
    
    # if  df['pris'].iloc[-2] < df['upper_band'].iloc[-2] and df['pris'].iloc[-1] > df['upper_band'].iloc[-1]:
    #     return 'sell'
    # elif df['pris'].iloc[-2] > df['lower_band'].iloc[-2] and df['pris'].iloc[-1] < df['lower_band'].iloc[-1]:
    #     return 'buy'
    # else:
    #     return 'hold'
    
    return df['b_strategy'].iloc[-1]


def ml_strategy(df, use_features, model):

    return # Bryt

    # Välj endast de features som används för prediction
    X = df[use_features].copy()


    # Gör predictions med modellen
    df['proba'] = model.predict_proba(X)[:,1]

    # df['buy_prob'] = probs[:, 1]
    # df['sell_prob'] = probs[:, 0]

    return df

def final_strategy(df, in_position):
    pass

def trading_logic():
    global df, use_features, model, in_psition
    # print('starting trading logic')
    
    #TODO: Lägg in en sell/buy/hold-kolumn från alla tre strategierna i df
    momentum_strategy(df)
    fibonacci_strategy(df)
    bollinger_strategy(df)
    # print('done with strategies')
    
    ml_strategy(df, use_features, model)
    # print('done with ml')
    
    # Här tar vi fram slutlig sell/buy/hold-strategi
    final_strategy(df, in_position)
    
    if True:
        print(df[['pris', 'sma_5', 'fib38', 'lower_band']].tail(1).values[0])
        
    
def handle_trading(out):
    global df
    out = pd.DataFrame({'pris':float(out['c'])},index=[pd.to_datetime(out['E'],unit='ms')])
    df = pd.concat([df,out],axis=0)
    trading_logic()

def on_message(ws, message):
    out = json.loads(message)
    handle_trading(out)

def on_open(ws):
    ws.send(the_message)

endpoint = "wss://stream.binance.com:9443/ws"
the_message = json.dumps({'method': 'SUBSCRIBE', 'params': ['btcusdt@ticker'], 'id': 1})
ws = websocket.WebSocketApp(endpoint, on_message=on_message, on_open=on_open)

#%%
def start():
    ws.run_forever()

if __name__ == "__main__":
    start()