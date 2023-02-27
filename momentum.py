#%%
import threading
import time
import websocket
import pandas as pd
import json

def init_fibonacci_strategy():
    pass
    
global df, in_position, use_features, model
df = pd.DataFrame()
in_position = False
model=None
use_features=[]
def get_m_strategy(row):
    current_price = row['close']
    sma_5 = row['sma_5']
    sma_20 = row['sma_20']

    # Om senaste priset är högre än sma_5 och sma_20,  "buy"
    if current_price > sma_5 and current_price > sma_20:
        return "buy"
    # Om senaste priset är lägre än sma_5 och sma_20,  "sell"
    elif current_price < sma_5 and current_price < sma_20:
        return "sell"
    # Annars "hold"
    else:
        return "hold"
        
        
def momentum_strategy(step=None):
    # print('momentum_strategy start')
    # Beräkna sma_5 och sma_20
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['m_strategy'] = 'hold' 
    # print('mom close2 step',step,'\n',df['close'].tail())
    
    
    df['m_strategy'] = df.apply(get_m_strategy, axis=1)
    # print('step', step, 'df.shape', df.shape, 'sista värdet', df[['m_strategy']].iloc[-1])
    # print('sista datum', df.tail(1).index)

    # print('SLUT','step',step, 'momentum_strategy momentum0:\n', df.tail(1)['m_strategy'].values,'   ', df.tail(1).index)
    # return df.tail(1)['m_strategy'].values[0]
    return df['m_strategy'].iloc[-1]

def get_f_strategy(row):
    print('get_f_strategy start',row.name)
    datum = row.name
    if 'entry' not in df.columns:
        # Sker första gången
        print('entry not in df.columns')
        start_a_new_fibonacci(df,datum)
        print('new columns',df.columns)
        return 'hold'
    
    # Köp när priset bryter över 'entry' nerifrån och upp
    if df['close'].shift(0).loc[datum] >= df['entry'].shift(0).loc[datum] and df['close'].shift(1).loc[datum] < df['entry'].shift(1).loc[datum]:
        return 'buy'
    # Säljsignal när priset bryter igenom target nerifrån och upp
    elif df['close'].shift(0).loc[datum] >= df['profit_target'].shift(0).loc[datum] and df['close'].shift(1).loc[datum] < df['profit_target'].shift(1).loc[datum]:
        return 'sell'
    # Säljsignal när priset bryter igenom stop_loss uppifrån och ner
    elif df['close'].shift(0).loc[datum] <= df['stop_loss'].shift(0).loc[datum] and df['close'].shift(1).loc[datum] > df['stop_loss'].shift(1).loc[datum]:
        return 'sell'
    # Behåll positionen annars
    else:
        return 'hold'


def start_a_new_fibonacci(df, datum):
    print('start_a_new_fibonacci', datum)

    if len(df) < 24:
        print('start_new', len(df), '< 24')
        df[['stop_loss', 'entry', 'profit_target']] = 0
        print('new columns',df.columns)
    high = df['close'].rolling(window=24).max()
    low = df['close'].rolling(window=24).min()

    diff = high - low
    print('high',high,'diff',diff)
    close = df.at[datum, 'close']
    levels = [close + level * diff for level in [-0.618, 0.618, 1.618]]
    print('levels',levels)


    df.loc[datum, 'stop_loss'] = df.loc[datum, 'close'] * levels[0]
    df.loc[datum, 'entry'] = df.loc[datum, 'close'] * levels[1]
    df.loc[datum, 'profit_target'] = df.loc[datum, 'close'] * levels[2]

    print('stop_loss', df.loc[datum, 'stop_loss'], 'entry', df.loc[datum, 'entry'], 'profit_target',
          df.loc[datum, 'profit_target'])

    df = df.fillna(0)
    return df

    
def fibonacci_strategy():
    global in_position, df
    print('fibonacci_strategy start')
    # vi köper när vi går över "entry" och in_position==False
    # vi säljer när vi är in_position och går under stop_loss eller över target
    # Vi räknar om stop_loss och target efter vi har sålt
    df['f_strategy'] = df.apply(get_f_strategy, axis=1)  
    if in_position and (df['f_strategy'].iloc[-1] == 'sell'):
        print("sell for", df['close'].iloc[-1])   
        start_a_new_fibonacci(df,df.iloc[-1].index)
        in_position=False
        print("return sell for", df['close'].iloc[-1])
        return 'sell'
        
    if not in_position and (df['f_strategy'].iloc[-1] == 'buy'):
        # borde egentligen vara nästa open
        print("return buy for", df['close'].iloc[-1])
        in_position = True
        return 'buy'

    print ('return hold and in_position =',in_position,'row =',df.iloc[-1].values)
    print(df.iloc[:,8:12])
    return 'hold'


def get_b_strategy(row):
    i = row.name        

    if df['close'].shift(1).loc[i] < df['upper_band'].shift(1).loc[i] and df['close'].shift(0).loc[i] > df['upper_band'].shift(0).loc[i]:
        return 'sell'
    elif df['close'].shift(1).loc[i] > df['lower_band'].shift(1).loc[i] and df['close'].shift(0).loc[i] < df['lower_band'].shift(0).loc[i]:
        return 'buy'
    else:
        return 'hold'

    
def bollinger_strategy():
    # print('bollinger_strategy start')
    # Beräkna 20-dagars rullande medelvärde och standardavvikelse
    df['boll_sma'] = df['close'].rolling(window=20).mean()
    df['boll_std'] = df['close'].rolling(window=20).std()

    # Beräkna övre och undre Bollinger-bands
    df['upper_band'] = df['boll_sma'] + (2 * df['boll_std'])
    df['lower_band'] = df['boll_sma'] - (2 * df['boll_std'])

    print(df.shape)

    df['b_strategy'] = df.apply(get_b_strategy, axis=1)

    if len(df) < 2:
        return 'hold'
    
    return df['b_strategy'].iloc[-1]


def ml_strategy(use_features, model):

    return # Bryt

    # Välj endast de features som används för prediction
    X = df[use_features].copy()


    # Gör predictions med modellen
    df['proba'] = model.predict_proba(X)[:,1]

    # df['buy_prob'] = probs[:, 1]
    # df['sell_prob'] = probs[:, 0]

    return df

def final_strategy(in_position_b, in_position_f, in_position_m):
    pass

def trading_logic():
    global df, use_features, model, in_position
    print('trading logic start')
    
    momentum_strategy()
    fibonacci_strategy()
    bollinger_strategy()
    # print('done with strategies')
    
    ml_strategy(use_features, model)
    # print('done with ml')
    
    # Här tar vi fram slutlig sell/buy/hold-strategi
    final_strategy(in_position, in_position, in_position)
    
    if False:
        print(df[['close', 'sma_5', 'entry', 'lower_band']].tail(1).values[0])
        
    
def handle_trading(out):
    global df
    open_price = float(out['o'])
    low_price = float(out['l'])
    high_price = float(out['h'])
    close_price = float(out['c'])
    volume = float(out['v'])
    
    out = pd.DataFrame({'open':float(out['o']),
                        'low':float(out['l']),
                        'high':float(out['h']),
                        'close':float(out['c']),
                        'volume':float(out['v']),
                        },
                       index=[pd.to_datetime(out['E'],unit='ms')])
    df = pd.concat([df,out],axis=0)
    trading_logic()

def on_message(ws, message):
    out = json.loads(message)
    handle_trading(out)

def on_open(ws):
    ws.send(the_message)


def stop():
    ws.close()

endpoint = "wss://stream.binance.com:9443/ws"
the_message = json.dumps({'method': 'SUBSCRIBE', 'params': ['ethusdt@ticker'], 'id': 1})
ws = websocket.WebSocketApp(endpoint, on_message=on_message, on_open=on_open)
#%%


def start(max_time=None):
    def stop():
        ws.keep_running = False

    def run_websocket():
        ws.run_forever()

    ws_thread = threading.Thread(target=run_websocket)
    ws_thread.start()

    if max_time is not None:
        timer = threading.Timer(max_time, stop)
        timer.start()

    ws_thread.join()


if __name__ == "__main__":
    start(max_time=30)

