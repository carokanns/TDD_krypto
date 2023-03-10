
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
import pytest
import websocket
import pandas as pd
import numpy as np
import json
import momentum as mom

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 120)

##############
# Test the handle_trading function
##############


def test_handle_trading():
    # Testa att lägga till en första rad med closedata till en tom dataframe
    out = {'e': '24hrTicker', 'E': 1676583178354, 's': 'BTCUSDT', 'p': '308.27000000', 'P': '1.278', 'w': '24645.97059023', 'x': '24113.92000000', \
        'c': '24421.49000000', 'Q': '0.01008000', 'b': '24421.49000000', 'B': '0.01067000', 'a': '24421.50000000', 'A': '0.04175000', \
        'o': '24113.22000000', 'h': '25250.00000000', 'l': '24057.03000000', 'v': '432518.55345000', 'q': '10659839548.05928360', 'O': 1676496778330, 'C': 1676583178330, 'F': 2709848802, 'L': 2720557328, 'n': 10708527}
    mom.handle_trading(out)
    expected_df = pd.DataFrame(
        {   'open':   [float(24113.22)],
            'high':   [float(25250.0)],
            'low':    [float(24057.03)],
            'close':  [float(24421.49)], 
            'volume': [float(432518.55345)],
            'price': [float(-1)],
            # 'sma_5': [np.nan], 'sma_20': [np.nan], 'm_strategy': ['hold'],
            # 'fib38':[np.nan], 'fib62':[np.nan],'f_strategy': ['hold'], 
            # 'boll_sma':[np.nan], 'boll_std':[np.nan],  'upper_band': [np.nan],  'lower_band': [np.nan], 'b_strategy': ['hold']
            }, 
        index=pd.to_datetime(['2023-02-16 21:32:58.354000']))
      
    pd.testing.assert_frame_equal(mom.df[['open', 'high', 'low', 'close', 'volume', 'price' ]].iloc[-1:], expected_df)

    # Testa att lägga till en till rad med closedata till dataframe
    out = {'e': '24hrTicker', 'E': 1676583179354, 's': 'BTCUSDT', 'p': '303.48000000', 'P': '1.258', 'w': '24645.97518999', 'x': '24117.37000000', 
           'c': '24420.83000000', 'Q': '0.00659000', 'b': '24419.96000000', 'B': '0.05000000', 'a': '24421.43000000',
           'A': '0.14633000', 
           'o': '24117.35000000', 'h': '25250.00000000', 'l': '24057.03000000', 'v': '432516.83593000', 'q': '10659799207.58276380', 'O': 1676496779309, 'C': 1676583179309, 'F': 2709848988, 'L': 2720557420, 'n': 10708433}
    mom.handle_trading(out)
    expected_df = pd.DataFrame({
                                'open':   [float(24113.22), float(24117.35)],
                                'high':   [float(25250.0) , float(25250.0)],
                                'low':    [float(24057.03), float(24057.03)],
                                'close':  [float(24421.49), float(24420.83)],
                                'volume': [float(432518.55345), float(432516.83593)],
                                 'price': [float(-1)],

                                # 'sma_5': [np.nan,np.nan], 'sma_20': [np.nan,np.nan],'m_strategy': ['hold','hold'],
                                # 'fib38': [np.nan,np.nan], 'fib62': [np.nan,np.nan], 'f_strategy': ['hold','hold'],
                                # 'boll_sma': [np.nan,np.nan], 'boll_std': [np.nan,np.nan], 'upper_band': [np.nan,np.nan],  'lower_band': [np.nan,np.nan],
                                # 'b_strategy': ['hold','hold']
                                }, 
                               index=pd.to_datetime(['2023-02-16 21:32:58.354000', '2023-02-16 21:32:59.354000']))

    print('df.shape',mom.df.shape)
    print('df.columns',mom.df.columns)
    print('expected.shape',expected_df.shape)
    print('expected.columns',expected_df.columns)
    print('df\n',mom.df.iloc[-2:])
    print('expected\n', expected_df)
    pd.testing.assert_frame_equal(
        mom.df[['open', 'high', 'low', 'close', 'volume', 'price']].iloc[-2:], expected_df)


def test_trading_logic():
    global df
    mom.use_features = ['price','close',  'sma_5',  'sma_20',  'fib38', 'fib62',  'boll_sma',  'boll_std',  'upper_band',  'lower_band']
    mom.df = pd.DataFrame({
                                'open':   [float(24113.22), float(24117.35)],
                                'high':   [float(25250.0) , float(25250.0)],
                                'low':    [float(24057.03), float(24057.03)],
                                'close':  [float(24421.49), float(24420.83)],
                                'volume': [float(432518.55345), float(432516.83593)],
                                 'price': [float(-1)],

                                # 'sma_5': [np.nan,np.nan], 'sma_20': [np.nan,np.nan],'m_strategy': ['hold','hold'],
                                # 'fib38': [np.nan,np.nan], 'fib62': [np.nan,np.nan], 'f_strategy': ['hold','hold'],
                                # 'boll_sma': [np.nan,np.nan], 'boll_std': [np.nan,np.nan], 'upper_band': [np.nan,np.nan],  'lower_band': [np.nan,np.nan],
                                # 'b_strategy': ['hold','hold']
                                }, 
                               index=pd.to_datetime(['2023-02-16 21:32:58.354000', '2023-02-16 21:32:59.354000']))

    mom.trading_logic()
    print('Efter df\n', mom.df.tail(3))
    assert mom.df.shape[0] == 2, f'Number of rows in df should be 2 but shape is {mom.df.shape}'

def test_momentum_strategy():
    # Skapa en dataframe med testdata
    test_prices = [10, 12,  8,  9, 11, 13, 14, 15, 17, 16, 
                   14, 13, 12, 11, 10, 11, 12, 12, 11, 10, 
                   11, 12, 10, 12,  8,  9, 11, 13, 14, 15, 
                   17, 16, 14, 13, 12, 11, 10, 11, 12, 12, 
                   11, 15, 17, 16]
    mom.df = pd.DataFrame({'close': test_prices}, index=pd.date_range(start='2022-01-01', periods=len(test_prices), freq='D'))
    # print('test_df 1', mom.df.tail(3))
    # Anropa momentum_strategy-funktionen
    result = mom.momentum_strategy(1)
    
    # print(test_df)  # skrivs ut om testet misslyckas
    
    # Kontrollera att resultatet är "buy"
    # print('step 1',mom.df[['close','sma_5','sma_20', 'm_strategy']])
    # print('momentum1b',mom.df.tail(1)['m_strategy'].values[0])
    # print('momentum2b',mom.df.iloc[-1]['m_strategy'])
    # print('momentum3b',result)
    assert result == "buy"
    assert mom.df.iloc[-1]['m_strategy']  == 'buy'
    
    # ta bort de 3 sista raderna skapar en sell-signal
    mom.df = mom.df.iloc[:-3]
    
    # Anropa momentum_strategy-funktionen 
    result = mom.momentum_strategy(2)
    # print('result dat step 2',mom.df.tail(1).index)
    # print(test_df) # skrivs ut om testet misslyckas
    
    # Kontrollera att resultatet är "sell"
    # print('step 2',mom.df[['close','sma_5','sma_20', 'm_strategy']].tail(10))
    # print('momentum1s',mom.df.tail(1)['m_strategy'].values[0])
    # print('momentum2s',mom.df.iloc[-1]['m_strategy'])
    # print('momentum3s',result)
    # print(mom.df.tail(10)['m_strategy'], '\nlen =', mom.df.shape)
    assert result == "sell" 
    assert mom.df.iloc[-1]['m_strategy']  == 'sell'

    # ändra sista close till 12
    mom.df.iloc[-1,0] = 12
    
    # Anropa momentum_strategy-funktionen
    result = mom.momentum_strategy(3)

    # print(test_df)  # skrivs ut om testet misslyckas

    # Kontrollera att resultatet är "hold"
    # print('step 3',mom.df[['close','sma_5','sma_20', 'm_strategy']])
    # print('momentum1h',mom.df.tail(1)['m_strategy'].values[0])
    # print('momentum2h',mom.df.iloc[-1]['m_strategy'])
    # print('momentum3h',result)
    assert result == "hold", 'result should be "hold"'
    assert mom.df.iloc[-1]['m_strategy'] == 'hold', 'last row should be "hold"'


def test_get_f_values():
    # Skapa en DataFrame med 3 rader och 4 kolumner
    np.random.seed(123)
    high = np.random.uniform(low=0.1, high=1.0, size=3)
    low = np.random.uniform(low=0.0, high=high, size=3)
    close = np.random.uniform(low=low, high=high, size=3)
    open = np.random.uniform(low=low, high=high, size=3)

    # Skapa DataFrame
    data = {'open':open, 'high': high, 'low': low, 'close': close}
    mom.df = pd.DataFrame(data)
    
    # Skapa en DateTimeIndex med tre datum/tider
    dates = pd.date_range('2022-01-01', periods=3, freq='D')
    mom.df.set_index(dates, inplace=True)
    print('df', mom.df)
    # Beräkna entry, stop_loss och profit_target
    stop_loss, entry, profit_target = mom.get_f_values(mom.df)
    print('entry', entry, 'stop_loss', stop_loss, 'profit_target', profit_target)
    # Kontrollera att entry, stop_loss och profit_target är int-värden
    assert isinstance(entry, float)
    assert isinstance(stop_loss, float)
    assert isinstance(profit_target, float)

    # Kontrollera att entry är större än stop_loss, som i sin tur är större än profit_target
    assert (entry > stop_loss) and  (entry < profit_target)

    # Kontrollera att entry, stop_loss och profit_target har rätt värden
    diff = mom.df.high[-1] - mom.df.low[-1]
    l = [-0.618*diff, 0.618*diff, 1.618*diff]
    assert stop_loss == mom.df.close[-1] + l[0]
    assert entry == mom.df.close[-1] + l[1], f'entry should be: {0.618*diff}, df.close[-1]: {mom.df.close[-1]}'
    assert profit_target == mom.df.close[-1] + l[2]


# Create a dummy dataframe with some test data
data = {'open': [1.0, 2.0, 3.0, 4.0, 5.0],
        'high': [2.0, 3.0, 4.0, 5.0, 6.0],
        'low': [1.0, 1.0, 2.0, 3.0, 4.0],
        'close': [2.0, 3.0, 4.0, 5.0, 6.0],
        'volume': [100, 200, 300, 400, 500]}
mom.df = pd.DataFrame(data, index=pd.date_range('2022-01-01', periods=5, freq='D'))


def test_fibonacci_strategy():
    # global stop_loss, entry, profit_target, buy_next, in_position
    mom.df = pd.DataFrame({'open': [100, 90, 95, 149], 'high': [110, 100, 105, 152],
                       'low': [90, 80, 85, 149], 'close': [105, 95, 100, 152], 'price': [-1, -1, -1, -1]},
                      index=pd.date_range(start='2022-01-01', periods=4, freq='D'))
    mom.stop_loss, mom.entry, mom.profit_target = mom.get_f_values(mom.df)
    mom.buy_next = False
    mom.in_position = None
    # First time function is called, in_position is None
    mom.fibonacci_strategy()
    assert mom.df['price'].iloc[-1] == -1, f'price should be -1 but is {mom.df["price"].iloc[-1]}'
    assert mom.in_position == False, f'in_position should be False but is {mom.in_position}'
    
    # Buy condition is met, buy_next should then be True
    mom.fibonacci_strategy()
    print('price',mom.df['price'].iloc[-1],'open',mom.df['open'].iloc[-1], 'close',mom.df['close'].iloc[-1])
    print('buy_next', mom.buy_next, 'in_position', mom.in_position, 'entry',
          mom.entry, 'stop_loss', mom.stop_loss, 'profit_target', mom.profit_target)
    
    # add one row to dataframe för att komma i köp-läge
    mom.df = pd.concat([mom.df,pd.DataFrame({'open': [149], 'high': [155],  'low': [149], 'close': [154], 'price': [-1]},
                        index=pd.date_range(start='2022-01-05', periods=1, freq='D'))])
    print('Före köpläge df\n', mom.df)
    print('Före köpläge: price',mom.df['price'].iloc[-1],'open',mom.df['open'].iloc[-1], 'close',mom.df['close'].iloc[-1])

    mom.fibonacci_strategy()
    assert mom.buy_next == True, f'buy_next should be True but is {mom.buy_next}'
    assert mom.df['price'].iloc[-1] == -1  
    
    # add one row to dataframe för att göra köpet
    print('Före köp df\n', mom.df)
    mom.df = pd.concat([mom.df,pd.DataFrame({'open': [120], 'high': [155],  'low': [120], 'close': [154], 'price': [-1]},
                        index=pd.date_range(start='2022-01-06', periods=1, freq='D'))])
    
    mom.fibonacci_strategy()
    print('Efter köp df\n', mom.df)
    print('Efter köp:', 'price', mom.df['price'].iloc[-1], 'open',
          mom.df['open'].iloc[-1], 'close', mom.df['close'].iloc[-1])

    assert mom.df['price'].iloc[-1] == mom.df['open'].iloc[-1]
    assert mom.buy_next == False, f'buy_next should be False but is {mom.buy_next}'
    assert mom.in_position == True, f'in_position should be True but is {mom.in_position}'
     
    # add one row to dataframe för att sälja
    mom.df = pd.concat([mom.df,pd.DataFrame({'open': [120], 'high': [160],  'low': [110], 'close': [160], 'price': [120]},
                        index=pd.date_range(start='2022-01-07', periods=1, freq='D'))])
    print('Före sälj df\n', mom.df)
    print('Före sälj: price',mom.df['price'].iloc[-1],'open',mom.df['open'].iloc[-1], 'close',mom.df['close'].iloc[-1])
    print('buy_next', mom.buy_next, 'in_position', mom.in_position, 'entry',
            mom.entry, 'stop_loss', mom.stop_loss, 'profit_target', mom.profit_target)    
            
    # close position since profit_target is reached, in_position should be False
    mom.fibonacci_strategy()
    print('Efter sälj df\n', mom.df)
    print('Efter sälj: price',mom.df['price'].iloc[-1],'open',mom.df['open'].iloc[-1], 'close',mom.df['close'].iloc[-1])
   
    assert mom.in_position == False
    assert mom.df['price'].iloc[-1] == mom.df['close'].iloc[-1]
    assert mom.buy_next == False, f'buy_next should be False but is {mom.buy_next}'

    # open new position, buy_next should then be True
    # add one row to dataframe för att få köpsignal
    mom.df = pd.concat([mom.df, pd.DataFrame({'open': [120], 'high': [160],  'low': [110], 'close': [160], 'price': [120]},
                        index=pd.date_range(start='2022-01-08', periods=1, freq='D'))])
    mom.fibonacci_strategy()
    assert mom.buy_next == True, f'buy_next should be True but is {mom.buy_next}'
    assert mom.in_position == False, f'in_position should be False but is {mom.in_position}'
    assert mom.df['price'].iloc[-1] == -1, f'price should be -1 but is {mom.df["price"].iloc[-1]}'
    
    # add one row to dataframe för att göra köpet
    mom.df = pd.concat([mom.df, pd.DataFrame({'open': [149], 'high': [155],  'low': [149], 'close': [154], 'price': [-1]},
                        index=pd.date_range(start='2022-01-09', periods=1, freq='D'))])
    mom.fibonacci_strategy()
    assert mom.in_position == True, f'in_position should be True but is {mom.in_position}'
    assert mom.df['price'].iloc[-1] == mom.df['open'].iloc[-1], f'price should be {mom.df["open"].iloc[-1]} but is {mom.df["price"].iloc[-1]}'
    
    # price falls below stop_loss, in_position should be False
    # add one row to dataframe för att trigga stop_loss
    mom.df = pd.concat([mom.df, pd.DataFrame({'open': [mom.stop_loss+5], 'high': [mom.stop_loss+10],  'low': [mom.stop_loss-2], 'close': [mom.stop_loss-1], 'price': [-1]},
                        index=pd.date_range(start='2022-01-09', periods=1, freq='D'))])
    mom.fibonacci_strategy()
    assert mom.in_position == False, f'in_position should be False but is {mom.in_position}'
    assert mom.df['price'].iloc[-1] == mom.df['close'].iloc[-1], f'price should be {mom.df["close"].iloc[-1]} but is {mom.df["price"].iloc[-1]}'
    
    # open new position, in_position should be True
    # add one row to dataframe för att få köpsignal
    mom.df = pd.concat([mom.df, pd.DataFrame({'open': [120], 'high': [160],  'low': [110], 'close': [160], 'price': [120]},
                        index=pd.date_range(start='2022-01-10', periods=1, freq='D'))])
    print('Före köpsignal: price',mom.df['price'].iloc[-1],'open',mom.df['open'].iloc[-1], 'close',mom.df['close'].iloc[-1])  
    mom.fibonacci_strategy()
    assert mom.buy_next == True
    assert mom.in_position == False
    assert mom.df['price'].iloc[-1] == -1  
    
    
    print('Före köp: price',mom.df['price'].iloc[-1],'open',mom.df['open'].iloc[-1], 'close',mom.df['close'].iloc[-1])
    mom.fibonacci_strategy()
    assert mom.in_position == True
    assert mom.buy_next == False
    assert mom.df['price'].iloc[-1] == mom.df['open'].iloc[-1]
    
    
    # price falls below stop_loss, in_position should be False
    mom.df = pd.concat([mom.df, pd.DataFrame({'open': [mom.stop_loss+5], 'high': [mom.stop_loss+10],  'low': [mom.stop_loss-2], 'close': [mom.stop_loss-1], 'price': [-1]},
                        index=pd.date_range(start='2022-01-11', periods=1, freq='D'))])
    mom.fibonacci_strategy()
    print('Fixa stop_loss')
    mom.fibonacci_strategy()
    assert mom.in_position == False
    assert mom.df['price'].iloc[-1] == mom.df[
        'close'].iloc[-1], f'stop_loss price should be {mom.df["close"].iloc[-1]} but is {mom.df["price"].iloc[-1]}'
    
    # price rises above entry, buy_next should be set to True
    mom.df = pd.concat([mom.df, pd.DataFrame({'open': [mom.entry-5], 'high': [mom.entry+10],  'low': [mom.entry-6], 'close': [mom.entry+1], 'price': [-1]},
                        index=pd.date_range(start='2022-01-12', periods=1, freq='D'))])
    mom.fibonacci_strategy()
    assert mom.in_position == False
    assert mom.buy_next == True
    assert mom.df['price'].iloc[-1] == -1, f'price should be -1 but is {mom.df["price"].iloc[-1]}'


def test_get_b_strategy():
    # Test "sell"
    mom.df = pd.DataFrame({'close': [100, 90, 95], 'upper_band': [
                      110, 100, 90], 'lower_band': [90, 80, 70]},
                          index=pd.date_range(start='2022-01-10', periods=3, freq='D'))
    mom.df['b_strategy'] = mom.df.apply(mom.get_b_strategy, axis=1)
    assert mom.df['b_strategy'].iloc[-1] == 'sell', f'b_strategy should be sell but is {mom.df["b_strategy"].iloc[-1]}'

    # Test "buy"
    mom.df = pd.DataFrame({'close': [80, 90, 85], 'upper_band': [
                      90, 100, 110], 'lower_band': [70, 80, 90]})
    mom.df['b_strategy'] = mom.df.apply(mom.get_b_strategy, axis=1)
    assert mom.df['b_strategy'].iloc[-1] == 'buy', f'b_strategy should be buy but is {mom.df["b_strategy"].iloc[-1]}'

    # Test "hold"
    mom.df = pd.DataFrame({'close': [100, 90, 80], 'upper_band': [
                      120, 110, 100], 'lower_band': [80, 70, 60]})
    mom.df['b_strategy'] = mom.df.apply(mom.get_b_strategy, axis=1)
    assert mom.df['b_strategy'].iloc[-1] == 'hold', f'b_strategy should be hold but is {mom.df["b_strategy"].iloc[-1]}'

def test_bollinger_strategy():
    test_data = [100, 110, 120, 130, 140, 150, 160, 150, 140, 130,
                 120, 110, 100, 90, 100, 110, 120, 130, 140, 130, 120, 110, 100]
    index = pd.date_range(start='2022-01-01', periods=len(test_data), freq='D')
    mom.df = pd.DataFrame({'close': test_data}, index=index)
    mom.in_position = None
    mom.bollinger_strategy()
    assert mom.in_position == None

    # Köp-signal
    test_data = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 99, 70]
    index = pd.date_range(start='2022-01-01', periods=len(test_data), freq='D')
    mom.df = pd.DataFrame({'close': test_data}, index=index)
    assert mom.bollinger_strategy() == 'buy'
    # print(mom.df[['close', 'upper_band', 'lower_band', 'b_strategy']].iloc[-2:])
    assert mom.df['b_strategy'].iloc[-1]  == 'buy'

    # Sälj-signal
    test_data = [200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100,
                 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 235]
    index = pd.date_range(start='2022-01-01', periods=len(test_data), freq='D')
    mom.df = pd.DataFrame({'close': test_data}, index=index)
    assert mom.bollinger_strategy() == 'sell'
    assert mom.df['b_strategy'].iloc[-1]  == 'sell'

    # Behåll-position
    test_data = [100, 90, 80, 70, 60, 50, 40, 50, 60, 70, 80, 90,
                 100, 110, 120, 130, 140, 150, 160, 150, 140, 130, 120, 110]
    index = pd.date_range(start='2022-01-01', periods=len(test_data), freq='D')
    mom.df = pd.DataFrame({'close': test_data}, index=index)
    assert mom.bollinger_strategy() == 'hold'
    assert mom.df['b_strategy'].iloc[-1]  == 'hold'
