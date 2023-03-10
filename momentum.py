# %%
import random
import threading
import time
import websocket
import pandas as pd
import json

global df, in_position, use_features, model
global entry, stop_loss, profit_target
df = pd.DataFrame()
in_position = None
model = None
entry, stop_loss, profit_target, buy_next = None, None, None, False
use_features = []


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
    global df
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


def get_f_values(df_):
    print('get_f_values start: df_.shape', df_.shape)
    diff = df.high[-1] - df.low[-1]
    l = [-0.618*diff, 0.618*diff, 1.618*diff]
    stop_loss, entry, profit_target = df_.close[-1] + \
        l[0], df_.close[-1]+l[1], df_.close[-1]+l[2]

    return stop_loss, entry, profit_target


def fibonacci_strategy():
    global df, stop_loss, entry, profit_target, buy_next, in_position
    print('fibonacci_strategy start: df.shape', df.shape)
    """
    Funktionen har globalerna df, stop_loss, entry, profit_target, buy_next, och in_position Den returnerar inga värden.

    Den kallas varje gång vi får en ny post som har lagts till sist i df med bl.a kolumnerna['open', 'high', 'low', 'close']
    Första gången som den kallas är in_position = None och då skall vi sätta in_position = False och kalla på funktionen som vi testade innan: stop_loss, entry, profit_target = get_f_values(df)df.price = -1 och göra return
    Därefter anvnäds hela tiden sista raden i df med:
    om buy_next == True:  Sätt in_position = True och df['price'] = df.open och buy_next = False, print('Buy for', df.open) och return

    om in_position == False och df.close > entry: sätt buy_next = True, sätt df.price = -1 och return
    om in_position == True: df.price = föregående rads df.price
    om in_position == True och(df.close > profit_target or df.close < stop_loss): kör stop_loss, entry, profit_target = get_f_values(df), sätt in_position = False och print('Sell for', df.close), gör return
    annars sätt df.price = -1 och return
    """
    last_row = df.index[-1]
    if in_position == None:
        print('Första gången: Vi initiera in_position = False och sätter df.price = -1')

        print('df.columns', df.columns, 'df', df.info())
        in_position = False
        buy_next = False
        stop_loss, entry, profit_target = get_f_values(df)

        print(df.index[-1])
        df.loc[last_row, 'price'] = -1
        return

    # print(df.loc[last_row,[ 'open', 'high', 'low', 'close', 'price']])
    print('in_position', in_position, 'entry', entry,
          'stop_loss', stop_loss, 'profit_target', profit_target)
    print(df)
    print('GT', df['close'].iloc[-1] > entry)
    if in_position == False and buy_next == False and df['close'].iloc[-1] > entry:
        print('Köpläge uppnått: Vi sätter buy_next till True')

        buy_next = True

        df.loc[last_row, 'price'] = -1

        return
    if buy_next == True:
        print('Dags för köp: Vi sätter buy_next till False och in_position till True och sätter df.price till df.open')

        buy_next = False
        df.loc[last_row, 'price'] = df['open'].iloc[-1]
        in_position = True
        return

    if in_position == True and df['close'].iloc[-1] > profit_target:
        print('Dags att sälja (profit_target): Vi sätter in_position till False och sätter df.price till close')

        buy_next = False
        in_position = False
        df.loc[last_row, 'price'] = df['close'].iloc[-1]

        return

    if in_position == True and df['close'].iloc[-1] < stop_loss:
        print('Dags att sälja (stop_loss): Vi sätter in_position till False och sätter df.price till close')

        buy_next = False
        in_position = False
        df.loc[last_row, 'price'] = df['close'].iloc[-1]

        return

    print('Ingen köp eller sälj signal denna gång: Vi sätter df.price till föregående price')
    df.loc[last_row, 'price'] = df['price'].iloc[-1]


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

    return  # Bryt

    # Välj endast de features som används för prediction
    X = df[use_features].copy()

    # Gör predictions med modellen
    df['proba'] = model.predict_proba(X)[:, 1]

    # df['buy_prob'] = probs[:, 1]
    # df['sell_prob'] = probs[:, 0]

    return df


def final_strategy(in_position_b, in_position_f, in_position_m):
    pass


def trading_logic():
    global df, use_features, model, in_position
    global entry, stop_loss, profit_target  # fibonacci
    print('trading logic start')

    # check if stop_loss has a value
    momentum_strategy()
    fibonacci_strategy()
    bollinger_strategy()
    # print('done with strategies')

    ml_strategy(use_features, model)
    # print('done with ml')

    # Här tar vi fram slutlig sell/buy/hold-strategi
    final_strategy(in_position, in_position, in_position)

    # if True:
    #     print(df[['close', 'sma_5', 'price', 'lower_band']].tail(1))


def handle_trading(out):
    global df
    open_price = float(out['o'])
    low_price = float(out['l'])
    high_price = float(out['h'])
    close_price = float(out['c'])
    volume = float(out['v'])

    out = pd.DataFrame({'open': float(out['o']),
                        'high': float(out['h']),
                        'low': float(out['l']),
                        'close': float(out['c']),
                        'volume': float(out['v']),
                        'price': -1.0,
                        },
                       index=[pd.to_datetime(out['E'], unit='ms')])
    df = pd.concat([df, out], axis=0)
    trading_logic()


def on_message(ws, message):
    out = json.loads(message)
    handle_trading(out)


def on_open(ws):
    ws.send(the_message)


def stop():
    ws.close()


endpoint = "wss://stream.binance.com:9443/ws"
the_message = json.dumps(
    {'method': 'SUBSCRIBE', 'params': ['ethusdt@ticker'], 'id': 1})
ws = websocket.WebSocketApp(endpoint, on_message=on_message, on_open=on_open)
# %%


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
