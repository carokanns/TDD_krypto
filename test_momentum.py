import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
import pytest
import websocket
import pandas as pd
import numpy as np
import json
import momentum as mom


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
            'low':    [float(24057.03)],
            'high':   [float(25250.0)],
            'close':  [float(24421.49)], 
            'volume': [float(432518.55345)],
            'sma_5': [np.nan], 'sma_20': [np.nan], 'm_strategy': ['hold'],
            'fib38':[np.nan], 'fib50':[np.nan], 'fib62':[np.nan],'f_strategy': ['hold'], 
            'boll_sma':[np.nan], 'boll_std':[np.nan],  'upper_band': [np.nan],  'lower_band': [np.nan], 'b_strategy': ['hold']}, 
        index=pd.to_datetime(['2023-02-16 21:32:58.354000']))
    
    print(mom.df.shape)
    print(mom.df.columns)
    print(expected_df.shape)
    print(expected_df.columns)
    
    pd.testing.assert_frame_equal(mom.df, expected_df)

    # Testa att lägga till en till rad med closedata till dataframe
    out = {'e': '24hrTicker', 'E': 1676583179354, 's': 'BTCUSDT', 'p': '303.48000000', 'P': '1.258', 'w': '24645.97518999', 'x': '24117.37000000', 
           'c': '24420.83000000', 'Q': '0.00659000', 'b': '24419.96000000', 'B': '0.05000000', 'a': '24421.43000000',
           'A': '0.14633000', 
           'o': '24117.35000000', 'h': '25250.00000000', 'l': '24057.03000000', 'v': '432516.83593000', 'q': '10659799207.58276380', 'O': 1676496779309, 'C': 1676583179309, 'F': 2709848988, 'L': 2720557420, 'n': 10708433}
    mom.handle_trading(out)
    expected_df = pd.DataFrame({
                                'open':   [float(24113.22), float(24117.35)],
                                'low':    [float(24057.03), float(24057.03)],
                                'high':   [float(25250.0) , float(25250.0)],
                                'close':  [float(24421.49), float(24420.83)],
                                'volume': [float(432518.55345), float(432516.83593)],
                                'sma_5': [np.nan,np.nan], 'sma_20': [np.nan,np.nan],'m_strategy': ['hold','hold'],
                                'fib38': [np.nan,np.nan], 'fib50': [np.nan,np.nan], 'fib62': [np.nan,np.nan], 'f_strategy': ['hold','hold'],
                                'boll_sma': [np.nan,np.nan], 'boll_std': [np.nan,np.nan], 'upper_band': [np.nan,np.nan],  'lower_band': [np.nan,np.nan],
                                'b_strategy': ['hold','hold']}, 
                               index=pd.to_datetime(['2023-02-16 21:32:58.354000', '2023-02-16 21:32:59.354000']))

    pd.testing.assert_frame_equal(mom.df, expected_df)


def test_trading_logic():
    global df
    mom.use_features = ['close',  'sma_5',  'sma_20',  'fib38',  'fib50', 'fib62',  'boll_sma',  'boll_std',  'upper_band',  'lower_band']
    
    data = [1, 2, 3, 4, 5]*10
    mom.df = pd.DataFrame({'close':data}, index=pd.date_range(
        start='2022-01-01', periods=len(data), freq='D'))

    # Run the trading logic
    mom.trading_logic()

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
    assert result == "hold"
    assert mom.df.iloc[-1]['m_strategy'] == 'hold'



def test_fibonacci_strategy():

    test_data = [100, 90, 110, 120, 130, 140, 150, 160, 170,
                 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70]

    test_data += [100, 90, 110, 120, 130, 140, 130, 120, 110, 100, 112, 113]*2

    mom.df = pd.DataFrame({'close': test_data}, index=pd.date_range(
        start='2022-01-01', periods=len(test_data), freq='D'))
    
    mom.fibonacci_strategy()
    
    assert mom.df.loc['2022-01-27'].fib38 == 137.98
    assert mom.df.loc['2022-01-28', 'fib38'] == 137.98
    assert mom.df.loc['2022-01-29', 'fib38'] == 137.98
    assert mom.df.loc['2022-01-30', 'fib38'] == 137.98
    assert mom.df.loc['2022-01-31', 'fib38'] == 131.80
    assert mom.df.loc['2022-02-01', 'fib62'] == 104.38
    assert mom.df.loc['2022-02-02', 'fib62'] == 100.56
    assert mom.df.loc['2022-02-03', 'fib62'] == 96.74000000000001
    assert mom.df.loc['2022-02-04', 'fib62'] == 96.74000000000001
    assert mom.df.loc['2022-02-05', 'fib62'] == 96.74000000000001

    assert mom.in_position == False

    # Test buy signal
    test_data = [80, 90, 80, 70, 80, 90, 100, 110, 93, 110]*4
    mom.df = pd.DataFrame({'close': test_data}, index=pd.date_range(
        start='2022-01-01', periods=len(test_data), freq='D'))
    assert mom.fibonacci_strategy() == 'buy'
    assert mom.df['f_strategy'].iloc[-1]  == 'buy'

    # Test sell signal
    test_data = [130, 120, 110, 100, 90, 80, 90, 100, 100, 80]*4
    mom.df = pd.DataFrame({'close': test_data}, index=pd.date_range(
        start='2022-01-01', periods=len(test_data), freq='D'))
    assert mom.fibonacci_strategy() == 'sell'
    assert mom.df['f_strategy'].iloc[-1]  == 'sell'

    # Test hold signal
    test_data = [100, 90, 80, 90, 80, 90, 80, 90, 80, 90]*4
    mom.df = pd.DataFrame({'close': test_data}, index=pd.date_range(
        start='2022-01-01', periods=len(test_data), freq='D'))
    assert mom.fibonacci_strategy() == 'hold'
    assert mom.df['f_strategy'].iloc[-1]  == 'hold'

def test_bollinger_strategy():
    test_data = [100, 110, 120, 130, 140, 150, 160, 150, 140, 130,
                 120, 110, 100, 90, 100, 110, 120, 130, 140, 130, 120, 110, 100]
    index = pd.date_range(start='2022-01-01', periods=len(test_data), freq='D')
    mom.df = pd.DataFrame({'close': test_data}, index=index)
    mom.bollinger_strategy()
    assert mom.in_position == False

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
