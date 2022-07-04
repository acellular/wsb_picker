from datetime import date, timedelta
import yfinance as yf

#get percentage stock change from open or close to open or close
def stock_change(stockHist, day, period='.5d', start='open', verbose=False):

    # potential accepted periods '1d', '.5d'-->for close open or open close?
    if (start == 'open'):
        if (period == '.5d'):

            if day.weekday() == 4: raise Exception('START DAY FRIDAY')
            elif day.weekday() == 5: raise Exception('START DAY SATURDAY')

            dayAfter = str(day+timedelta(1))

            diff = stockHist.loc[dayAfter][3] - stockHist.loc[dayAfter][0]
            diff = diff / stockHist.loc[dayAfter][0]  # percentage

    elif (start == 'close'):
        if (period == '.5d'):

            if day.weekday() == 4: raise Exception('START DAY FRIDAY')
            elif day.weekday() == 5: raise Exception('START DAY SATURDAY')

            dayAfter = str(day+timedelta(1))
            day_iloc = stockHist.index.get_loc(str(dayAfter)) - 1

            diff = stockHist.loc[dayAfter][0] - stockHist.iloc[day_iloc][3]
            diff = diff / stockHist.iloc[day_iloc][3]  # percentage
    if (start == 'sameday'):
        if (period == '.5d'):

            if day.weekday() == 4: raise Exception('START DAY FRIDAY')
            elif day.weekday() == 5: raise Exception('START DAY SATURDAY')

            dayAfter = str(day+timedelta(1))
            day_iloc = stockHist.index.get_loc(str(dayAfter)) - 1

            diff = stockHist.iloc[day_iloc][3] - stockHist.iloc[day_iloc][0]
            diff = diff / stockHist.iloc[day_iloc][0]  # percentage


    if verbose: print (f'Stock change: {diff}')
    return diff
    #TODO-other periods

#get stock history from yfinance
def stockHistory(stock, startday, interval='1d', end=None, verbose=False):
    stock = yf.Ticker(stock)
    if end is None:
        history = stock.history(start=startday, interval=interval)
    else:
        history = stock.history(start=startday, end=end, interval=interval)
    if verbose: print(history)
    return history


# TEST
if __name__ == "__main__":
    day = date.today()-timedelta(2)
    print (day)
    #SP = stockHistory('^GSPC', day-timedelta(45), verbose=True)#'^GSPC'
    #print(stock_change(SP, day, start='open'))
    stockHist = stockHistory('BB', day-timedelta(45), verbose=True)
    print(stock_change(stockHist, day, start='close'))

    #import Indicators
    #indicators, stockIndicts = Indicators.get_standard_indicators(stockHist, day)
    #print (indicators)
    #print (stockIndicts)
    #print (stockHistory('^VIX', day))
    #print(stockHist.iloc[4])

