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
    # global df  # Tillgång till den globala df

    # Testa att lägga till en första rad med prisdata till en tom dataframe
    out = {'e': '24hrTicker', 'E': 1676583178354, 's': 'BTCUSDT', 'p': '308.27000000', 'P': '1.278', 'w': '24645.97059023', 'x': '24113.92000000', \
        'c': '24421.49000000', 'Q': '0.01008000', 'b': '24421.49000000', 'B': '0.01067000', 'a': '24421.50000000', 'A': '0.04175000', 'o': '24113.22000000', 'h': '25250.00000000', 'l': '24057.03000000', 'v': '432518.55345000', 'q': '10659839548.05928360', 'O': 1676496778330, 'C': 1676583178330, 'F': 2709848802, 'L': 2720557328, 'n': 10708527}
    mom.handle_trading(out)
    expected_df = pd.DataFrame(
        {'pris': [24421.49]}, index=pd.to_datetime(['2023-02-16 21:32:58.354000']))
    
    # print('df\n',mom.df)
    # print('expected\n',expected_df)
    
    pd.testing.assert_frame_equal(mom.df, expected_df)

    # Testa att lägga till en till rad med prisdata till dataframe
    out = {'e': '24hrTicker', 'E': 1676583179354, 's': 'BTCUSDT', 'p': '303.48000000', 'P': '1.258', 'w': '24645.97518999', 'x': '24117.37000000', 
           'c': '24420.83000000', 'Q': '0.00659000', 'b': '24419.96000000', 'B': '0.05000000', 'a': '24421.43000000',
           'A': '0.14633000', 'o': '24117.35000000', 'h': '25250.00000000', 'l': '24057.03000000', 'v': '432516.83593000', 'q': '10659799207.58276380', 'O': 1676496779309, 'C': 1676583179309, 'F': 2709848988, 'L': 2720557420, 'n': 10708433}
    mom.handle_trading(out)
    expected_df = pd.DataFrame({'pris': [24421.49000000, 24420.83]}, index=pd.to_datetime(
        ['2023-02-16 21:32:58.354000', '2023-02-16 21:32:59.354000']))
    
    # print('df\n',mom.df)
    # print('expected\n',expected_df)

    pd.testing.assert_frame_equal(mom.df, expected_df)


def test_trading_logic():
    global df

    # Add some sample data to the dataframe
    data = {'pris': [1, 2, 3, 4, 5]}
    mom.df = pd.DataFrame(data, index=pd.date_range(
        start='2022-01-01', periods=5, freq='D'))

    # Run the trading logic
    mom.trading_logic()

    # Check that the sma_5 column has been added to the dataframe
    assert 'sma_5' in mom.df.columns

    # Check that the calculated SMA values are correct
    expected_sma_5 = [np.nan, np.nan, np.nan, np.nan, 3]
    assert np.allclose(mom.df['sma_5'].values, expected_sma_5, equal_nan=True)


def test_momentum_strategy():
    # Skapa en dataframe med testdata
    test_prices = [10, 12, 8, 9, 11, 13, 14,
                   15, 17, 16, 14, 13, 12, 11, 10, 11, 12, 12, 11, 10, 11, 12, 10, 12, 8, 9, 11, 13, 14,
                   15, 17, 16, 14, 13, 12, 11, 10, 11, 12, 12, 11, 15, 17, 16,]
    test_df = pd.DataFrame({'pris': test_prices}, index=pd.date_range(start='2022-01-01', periods=len(test_prices), freq='D'))
    
    # Anropa momentum_strategy-funktionen
    result = mom.momentum_strategy(test_df)
    
    print(test_df)  # skrivs ut om testet misslyckas
    
    # Kontrollera att resultatet är "buy"
    assert result == "buy"

    # ta bort de 3 sista raderna skapar en sell-signal
    test_df = test_df.iloc[:-3]
    
    # Anropa momentum_strategy-funktionen
    result = mom.momentum_strategy(test_df)
    
    print(test_df) # skrivs ut om testet misslyckas
    
    # Kontrollera att resultatet är "buy"
    assert result == "sell" 

    # ändra sista pris till 12
    test_df.iloc[-1,0] = 12
    
    # Anropa momentum_strategy-funktionen
    result = mom.momentum_strategy(test_df)

    print(test_df)  # skrivs ut om testet misslyckas

    # Kontrollera att resultatet är "buy"
    assert result == "hold"


def test_fibonacci_strategy():
    # global df

    test_data = [100, 90, 110, 120, 130, 140, 150, 160, 170,
                 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70]

    test_data += [100, 90, 110, 120, 130, 140, 130, 120, 110, 100, 112, 113]*2

    mom.df = pd.DataFrame({'pris': test_data}, index=pd.date_range(
        start='2022-01-01', periods=len(test_data), freq='D'))
    
    mom.fibonacci_strategy(mom.df)
    
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
    mom.df = pd.DataFrame({'pris': test_data}, index=pd.date_range(
        start='2022-01-01', periods=len(test_data), freq='D'))
    assert mom.fibonacci_strategy(mom.df) == 'buy'

    # Test sell signal
    test_data = [130, 120, 110, 100, 90, 80, 90, 100, 100, 80]*4
    mom.df = pd.DataFrame({'pris': test_data}, index=pd.date_range(
        start='2022-01-01', periods=len(test_data), freq='D'))
    assert mom.fibonacci_strategy(mom.df) == 'sell'

    # Test hold signal
    test_data = [100, 90, 80, 90, 80, 90, 80, 90, 80, 90]*4
    mom.df = pd.DataFrame({'pris': test_data}, index=pd.date_range(
        start='2022-01-01', periods=len(test_data), freq='D'))
    assert mom.fibonacci_strategy(mom.df) == 'hold'

def test_bollinger_strategy():
    test_data = {'pris': [100, 110, 120, 130, 125, 135, 145, 140, 130, 120, 110, 100, 95, 85, 90, 100, 110, 120, 130, 125]}
    test_df = pd.DataFrame(test_data, index=pd.date_range(start='2022-01-01', periods=len(test_data), freq='D'))
    mom.df = mom.add_bollinger_bands(test_df)
    
    assert mom.bollinger_strategy(mom.df) == 'sell'

    test_data = {'pris': [100, 110, 120, 130, 125, 135, 145, 140, 130, 120, 110, 100, 95, 85, 90, 100, 110, 120, 130, 135]}
    test_df = pd.DataFrame(test_data, index=pd.date_range(start='2022-01-01', periods=len(test_data), freq='D'))
    mom.df = mom.add_bollinger_bands(test_df)
    
    assert mom.bollinger_strategy(mom.df) == 'buy'

    test_data = {'pris': [100, 110, 120, 130, 125, 135, 145, 140, 130, 120, 110, 100, 95, 85, 90, 100, 110, 120, 130, 125]}
    test_df = pd.DataFrame(test_data, index=pd.date_range(start='2022-01-01', periods=len(test_data), freq='D'))
    mom.df = mom.add_bollinger_bands(test_df)
    
    assert mom.bollinger_strategy(mom.df) == 'hold'


def test_bollinger_strategy():
    test_data = [100, 110, 120, 130, 140, 150, 160, 150, 140, 130,
                 120, 110, 100, 90, 100, 110, 120, 130, 140, 130, 120, 110, 100]
    index = pd.date_range(start='2022-01-01', periods=len(test_data), freq='D')
    mom.df = pd.DataFrame({'pris': test_data}, index=index)
    mom.bollinger_strategy(mom.df)
    assert mom.in_position == False

    # Köp-signal
    test_data = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 99, 70]
    index = pd.date_range(start='2022-01-01', periods=len(test_data), freq='D')
    mom.df = pd.DataFrame({'pris': test_data}, index=index)
    assert mom.bollinger_strategy(mom.df) == 'buy'

    # Sälj-signal
    test_data = [200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100,
                 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 235]
    index = pd.date_range(start='2022-01-01', periods=len(test_data), freq='D')
    mom.df = pd.DataFrame({'pris': test_data}, index=index)
    assert mom.bollinger_strategy(mom.df) == 'sell'

    # Behåll-position
    test_data = [100, 90, 80, 70, 60, 50, 40, 50, 60, 70, 80, 90,
                 100, 110, 120, 130, 140, 150, 160, 150, 140, 130, 120, 110]
    index = pd.date_range(start='2022-01-01', periods=len(test_data), freq='D')
    mom.df = pd.DataFrame({'pris': test_data}, index=index)
    assert mom.bollinger_strategy(mom.df) == 'hold'
