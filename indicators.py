import numpy as np
import pandas as pd
# import ray.dataframe as pd


class Instrument:

    def __init__(self, df):
        self.odf = df
        self.df = df
        self._validate_df()

    ohlc = {'open', 'high', 'low', 'close'}

    UPTREND_CONTINUAL = 0
    UPTREND_REVERSAL = 1
    DOWNTREND_CONTINUAL = 2
    DOWNTREND_REVERSAL = 3

    def _validate_df(self):
        if not self.ohlc.issubset(self.df.columns):
            raise ValueError('DataFrame should have OHLC {} columns'.format(self.ohlc))


class Renko(Instrument):

    PERIOD_CLOSE = 1
    PRICE_MOVEMENT = 2

    TREND_CHANGE_DIFF = 2

    brick_size = 1
    chart_type = PERIOD_CLOSE

    def get_ohlc_data(self):
        if self.chart_type == self.PERIOD_CLOSE:
            self.period_close_bricks()
        else:
            self.price_movement_bricks()

        return self.cdf

    def price_movement_bricks(self):
        pass

    def period_close_bricks(self):
        brick_size = self.brick_size
        columns = ['Time', 'open', 'high', 'low', 'close']
        self.df = self.df[columns]

        self.cdf = pd.DataFrame(
            columns=columns,
            data=[],
        )

        self.cdf.loc[0] = self.df.loc[0]
        close = self.df.loc[0]['close'] // brick_size * brick_size
        self.cdf.iloc[0, 1:] = [close - brick_size, close, close - brick_size, close]
        self.cdf['uptrend'] = True

        columns = ['Time', 'open', 'high', 'low', 'close', 'uptrend']

        for index, row in self.df.iterrows():

            close = row['close']
            date = row['Time']

            row_p1 = self.cdf.iloc[-1]
            uptrend = row_p1['uptrend']
            close_p1 = row_p1['close']

            bricks = int((close - close_p1) / brick_size)
            data = []

            if uptrend and bricks >= 1:
                for i in range(bricks):
                    r = [date, close_p1, close_p1 + brick_size, close_p1, close_p1 + brick_size, uptrend]
                    data.append(r)
                    close_p1 += brick_size
            elif uptrend and bricks <= -2:
                uptrend = not uptrend
                bricks += 1
                close_p1 -= brick_size
                for i in range(abs(bricks)):
                    r = [date, close_p1, close_p1, close_p1 - brick_size, close_p1 - brick_size, uptrend]
                    data.append(r)
                    close_p1 -= brick_size
            elif not uptrend and bricks <= -1:
                for i in range(abs(bricks)):
                    r = [date, close_p1, close_p1, close_p1 - brick_size, close_p1 - brick_size, uptrend]
                    data.append(r)
                    close_p1 -= brick_size
            elif not uptrend and bricks >= 2:
                uptrend = not uptrend
                bricks -= 1
                close_p1 += brick_size
                for i in range(abs(bricks)):
                    r = [date, close_p1, close_p1 + brick_size, close_p1, close_p1 + brick_size, uptrend]
                    data.append(r)
                    close_p1 += brick_size
            else:
                continue

            sdf = pd.DataFrame(data=data, columns=columns)
            self.cdf = pd.concat([self.cdf, sdf])

        self.cdf.reset_index(inplace=True, drop=True)
        return self.cdf

    def shift_bricks(self):
        shift = self.df['close'].iloc[-1] - self.bdf['close'].iloc[-1]
        if abs(shift) < self.brick_size:
            return
        step = shift // self.brick_size
        self.bdf[['open', 'close']] += step * self.brick_size


class LineBreak(Instrument):

    line_number = 3

    def uptrend_reversal(self, close):
        lows = [self.cdf.iloc[i]['low'] for i in range(-1, -self.line_number - 1, -1)]
        least = min(lows)
        return close < least

    def downtrend_reversal(self, close):
        highs = [self.cdf.iloc[i]['high'] for i in range(-1, -self.line_number - 1, -1)]
        highest = max(highs)
        return close > highest

    def get_ohlc_data(self):
        columns = ['Time', 'open', 'high', 'low', 'close']
        self.df = self.df[columns]

        self.cdf = pd.DataFrame(columns=columns, data=[])

        for i in range(self.line_number):
            self.cdf.loc[i] = self.df.loc[i]

        self.cdf['uptrend'] = True

        columns = ['Time', 'open', 'high', 'low', 'close', 'uptrend']

        for index, row in self.df.iterrows():

            close = row['close']

            row_p1 = self.cdf.iloc[-1]

            uptrend = row_p1['uptrend']

            open_p1 = row_p1['open']
            close_p1 = row_p1['close']

            if uptrend and close > close_p1:
                r = [close_p1, close, close_p1, close]
            elif uptrend and self.uptrend_reversal(close):
                uptrend = not uptrend
                r = [open_p1, open_p1, close, close]
            elif not uptrend and close < close_p1:
                r = [close_p1, close_p1, close, close]
            elif not uptrend and self.downtrend_reversal(close):
                uptrend = not uptrend
                r = [open_p1, close, open_p1, close]
            else:
                continue

            sdf = pd.DataFrame(data=[[row['Time']] + r + [uptrend]], columns=columns)
            self.cdf = pd.concat([self.cdf, sdf])

        self.cdf.reset_index(inplace=True)
        return self.cdf


class PnF(Instrument):
    box_size = 2
    reversal_size = 3

    @property
    def brick_size(self):
        return self.box_size

    def get_state(self, uptrend_p1, bricks):
        state = None
        if uptrend_p1 and bricks > 0:
            state = self.UPTREND_CONTINUAL
        elif uptrend_p1 and bricks * -1 >= self.reversal_size:
            state = self.UPTREND_REVERSAL
        elif not uptrend_p1 and bricks < 0:
            state = self.DOWNTREND_CONTINUAL
        elif not uptrend_p1 and bricks >= self.reversal_size:
            state = self.DOWNTREND_REVERSAL
        return state

    def roundit(self, x, base=5):
        return int(base * round(float(x)/base))

    def get_ohlc_data(self, source='close'):
        source = source.lower()
        box_size = self.box_size
        data = self.df.itertuples()

        uptrend_p1 = True
        if source == 'close':
            open_ = self.df.iloc[0]['open']
            close = self.roundit(open_, base=self.box_size)
            pnf_data = [[0, 0, 0, 0, close, True]]
        else:
            low = self.df.iloc[0]['low']
            open_ = self.roundit(low, base=self.box_size)
            pnf_data = [[0, 0, open_, open_, open_, True]]

        for row in data:
            date = row.date
            close = row.close

            open_p1 = pnf_data[-1][1]
            high_p1 = pnf_data[-1][2]
            low_p1 = pnf_data[-1][3]
            close_p1 = pnf_data[-1][4]

            if source == 'close':
                bricks = int((close - close_p1) / box_size)
            elif source == 'hl':
                if uptrend_p1:
                    bricks = int((row.high - high_p1) / box_size)
                else:
                    bricks = int((row.low - low_p1) / box_size)
            state = self.get_state(uptrend_p1, bricks)

            if state is None:
                continue

            day_data = []

            if state == self.UPTREND_CONTINUAL:
                for i in range(bricks):
                    r = [date, close_p1, close_p1 + box_size, close_p1, close_p1 + box_size, uptrend_p1]
                    day_data.append(r)
                    close_p1 += box_size
            elif state == self.UPTREND_REVERSAL:
                uptrend_p1 = not uptrend_p1
                bricks += 1
                close_p1 -= box_size
                for i in range(abs(bricks)):
                    r = [date, close_p1, close_p1, close_p1 - box_size, close_p1 - box_size, uptrend_p1]
                    day_data.append(r)
                    close_p1 -= box_size
            elif state == self.DOWNTREND_CONTINUAL:
                for i in range(abs(bricks)):
                    r = [date, close_p1, close_p1, close_p1 - box_size, close_p1 - box_size, uptrend_p1]
                    day_data.append(r)
                    close_p1 -= box_size
            elif state == self.DOWNTREND_REVERSAL:
                uptrend_p1 = not uptrend_p1
                bricks -= 1
                close_p1 += box_size
                for i in range(abs(bricks)):
                    r = [date, close_p1, close_p1 + box_size, close_p1, close_p1 + box_size, uptrend_p1]
                    day_data.append(r)
                    close_p1 += box_size

            pnf_data.extend(day_data)
        self.cdf = pd.DataFrame(pnf_data[1:])
        self.cdf.columns = ['date', 'open', 'high', 'low', 'close', 'uptrend']
        return self.cdf

    def get_bar_ohlc_data(self, source='close'):
        df = self.get_ohlc_data(source=source)

        df['trend_change'] = df['uptrend'].ne(df['uptrend'].shift().bfill()).astype(int)
        df['trend_change_-1'] = df['trend_change'].shift(-1)

        start = df.iloc[0].values
        df = df[(df['trend_change'] == 1) | (df['trend_change_-1'] == 1)]
        data = np.vstack([start, df.values])
        df = pd.DataFrame(data)
        df.columns = ['date', 'open', 'high', 'low', 'close', 'uptrend', 'tc', 'tc1']

        bopen = df[['date', 'open']][df.index%2 == 0]
        bclose = df[['date', 'close']][df.index%2 == 1]

        bopen.reset_index(inplace=True, drop=True)
        bclose.reset_index(inplace=True, drop=True)
        bopen['close'] = bclose['close']
        df = bopen

        df['high'] = df[['open', 'close']].max(axis=1)
        df['low'] = df[['open', 'close']].min(axis=1)
        df.dropna(inplace=True)
        df[['open', 'close']] = df[['open', 'close']].astype(float)
        return df

def EMA(df, period, column='close'):
    return df[column].ewm(span=period, adjust=False).mean()

# Function to calculate MACD and MACD Histogram
def MACD(df, short_period=12, long_period=26, signal_period=9):
    short_ema = EMA(df, short_period)
    long_ema = EMA(df, long_period)
    df['MACD'] = short_ema - long_ema
    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    return df

# Function to implement the Elder Impulse System
def elder_impulse_system(df, ema_period=13):
    df['EMA'] = EMA(df, ema_period)
    df = MACD(df)
    conditions = [
        (df['close'] > df['EMA']) & (df['MACD_Histogram'] > 0),  # Buy condition
        (df['close'] < df['EMA']) & (df['MACD_Histogram'] < 0),  # Sell condition
    ]
    choices = ['Buy', 'Sell']
    df['Impulse'] = np.select(conditions, choices, default='Hold')
    return df

def calculate_heikin_ashi(df):
    df['HA_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['HA_Open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
    # For the first row, HA_Open should be the same as the first open
    df.at[0, 'HA_Open'] = (df.at[0, 'open'] + df.at[0, 'close']) / 2
    df['HA_High'] = df[['high', 'HA_Open', 'HA_close']].max(axis=1)
    df['HA_Low'] = df[['low', 'HA_Open', 'HA_close']].min(axis=1)
    # Calculate uptrend
    df['uptrend'] = df['HA_close'] > df['HA_Open']
    return df

def calc_choppiness_index(df, period=14):
    """
    Calculate Choppiness Index for a DataFrame containing OHLC values.

    Args:
        df (pd.DataFrame): DataFrame with OHLC values.
        period (int): Look-back period for calculating the Choppiness Index.

    Returns:
        pd.DataFrame: DataFrame with the Choppiness Index values added.
    """
    df['tr'] = df['high'].rolling(period).max() - df['low'].rolling(period).min()
    df['atr'] = df['tr'].rolling(period).mean()
    high_minus_low = df['high'] - df['low']
    high_minus_close = (df['high'] - df['close']).abs()
    low_minus_close = (df['low'] - df['close']).abs()
    df['numerator'] = high_minus_low.rolling(period).sum()
    df['denominator'] = df['atr'].rolling(period).sum()
    df['choppiness_index'] = 100 * (df['numerator'] / df['denominator']) / period
    df = df.drop(['tr', 'atr', 'numerator', 'denominator'], axis=1)
    return df

def calculate_adx(df, period=14):
    """
    Calculate Average Directional Index (ADX) for a DataFrame containing OHLC values.

    Args:
        df (pd.DataFrame): DataFrame with OHLC values.
        period (int): Look-back period for calculating ADX.

    Returns:
        pd.DataFrame: DataFrame with ADX values added.
    """
    # Calculate True Range (TR)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
    df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    # Calculate Directional Movement (DM)
    df['high_shifted'] = df['high'].shift(1)
    df['low_shifted'] = df['low'].shift(1)
    df['plus_dm'] = 0.0
    df['minus_dm'] = 0.0
    df.loc[(df['high'] - df['high_shifted'] > df['low_shifted'] - df['low']), 'plus_dm'] = df['high'] - df['high_shifted']
    df.loc[(df['low_shifted'] - df['low'] > df['high'] - df['high_shifted']), 'minus_dm'] = df['low_shifted'] - df['low']
    # Calculate True Directional Index (TRDI) and Directional Index (DI)
    df['atr'] = df['tr'].rolling(period).mean()
    df['plus_di'] = 100 * (df['plus_dm'].rolling(period).sum() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(period).sum() / df['atr'])
    # Calculate Directional Movement Index (DX)
    df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
    # Calculate ADX
    df['adx'] = df['dx'].rolling(period).mean()
    df = df.drop(['tr1', 'tr2', 'tr3', 'tr', 'high_shifted', 'low_shifted', 'plus_dm', 'minus_dm', 'atr', 'dx'], axis=1)
    return df

def calculate_rsi(data, period=14):
    """
    Calculate the Relative Strength Index (RSI) from candlestick data.

    Parameters:
        data (DataFrame): Candlestick data with 'time', 'open', 'high', 'low', 'close' columns.
        period (int): The period to consider for RSI calculation (default is 14).

    Returns:
        DataFrame: Original dataframe with 'rsi' column appended.
    """
    # Calculate price changes
    delta = data['close'].diff()
    # Get positive and negative price changes
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    # Calculate RS and RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # Add RSI column to the dataframe
    data['rsi'] = rsi
    return data

def calculate_ATR(df,period=14):
    # Calculate True Range (TR)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
    df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    # Calculate ATR
    df['ATR'] = df['tr'].rolling(period).mean()
    df = df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1)
    return df

def calculate_awesome_oscillator(data):
    """
    Calculate the Awesome Oscillator (AO).

    Parameters:
    data (pd.DataFrame): DataFrame with columns 'High' and 'Low'.

    Returns:
    pd.DataFrame: DataFrame with the original data and AO column.
    """
    # Calculate the Median Price
    data['Median_Price'] = (data['high'] + data['low']) / 2
    # Calculate the 5-period and 34-period simple moving averages
    data['SMA_5'] = data['Median_Price'].rolling(window=5).mean()
    data['SMA_34'] = data['Median_Price'].rolling(window=34).mean()
    # Calculate the Awesome Oscillator
    data['AO'] = data['SMA_5'] - data['SMA_34']
    return data

def renko_candle(df,bs):
  try:
    stock_prices=df
    df = calculate_ATR(df)
    if bs == "ATR/2":
      bs = int(int(df.ATR.iloc[-1])/2)
    elif bs == "ATR/4":
      bs = int(int(df.ATR.iloc[-1])/4)
    elif bs == "ATR/8":
      bs = int(int(df.ATR.iloc[-1])/8)
    else:pass

    if bs == 0:
      bs = 1
    renko = indicators.Renko(stock_prices)
    # set brick size
    renko.brick_size = bs
    data = renko.get_ohlc_data()
    data['Time'] = pd.to_datetime(data['Time'])
    stock_prices = data.set_index('Time')
    stock_prices = calculate_rsi(stock_prices,14)
    stock_prices = calculate_awesome_oscillator(stock_prices)
    stock_prices = elder_impulse_system(stock_prices,13)

    return stock_prices,df
  except Exception as e:
    print(f"Error with Renko : {e}")
    pass