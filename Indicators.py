from datetime import date, datetime, timedelta
import time
import statistics
#import hurst


import StockChecker

#TODO--print to file or use indicators to stop incomplete making it to file somehow??

def moving_average(stockHist, day, period=10):
    
    start_day_index = stockHist.index.get_loc(str(day))

    #average the closing price for given period past
    total = 0
    for i in range(period):
        #print (stockHist.iloc[start_day_index-i][3])
        try:
            total += stockHist.iloc[start_day_index-i][3]
        except:
            print('problem getting stock data out, MA, prob key error, not full data')
            return -1

    return total/period


def CBOE_volatility(stockHist, day):#DONE IN MAIN NOW SINCE OTHERWISE CONSTANTLY GRABBING
    return StockChecker.stockHistory('^VIX', day).iloc[0][3]
    #TODOshould really be added elsewhere so not repeat I guess
    #not sure really worth it since will be same for day, unless some amazing poly combo?


#OTHER SIMPLE INDICATORS
def average_volume(stockHist, day, period=10):
    start_day_index = stockHist.index.get_loc(str(day))
    total = 0
    for i in range(period):

        try:
            total += stockHist.iloc[start_day_index-i][4]
        except:
            print('problem getting stock data out, AV-VOL, prob key error, not full data')
            return -1

    return total/period

def price_stdev(stockHist, day, period=20):
    start_day_index = stockHist.index.get_loc(str(day))

    closes = []
    for i in range(period):
        try:
            closes.append(stockHist.iloc[start_day_index-i][3])
        except:
            print('problem getting stock data out, P-STDEV, prob key error, not full data')
            return -1

    return statistics.stdev(closes)

def volume_stdev(stockHist, day, period=20):#for closes only???
    start_day_index = stockHist.index.get_loc(str(day))

    volumes = []
    for i in range(period):
        try:
            volumes.append(stockHist.iloc[start_day_index-i][4])
        except:
            print('problem getting stock data out, V-STDEV, prob key error, not full data')
            return -1

    return statistics.stdev(volumes)

"""
def price_hurst(stockHist, day, period=10):#OR WILL USE HURST FOR THE ABOVE??-->just use the availible library?
    start_day_index = stockHist.index.get_loc(str(day))

    #average the closing price for given period past
    closes = []
    for i in range(period):
        #print (stockHist.iloc[start_day_index-i][3])
        try:
            closes.append(stockHist.iloc[start_day_index-i][3])
        except:
            print('problem getting stock data out, P-HURST, prob key error, not full data')
            return -1
    H, c, data = hurst.compute_Hc(closes, kind='price', simplified=True)
    return H

def volume_hurst(stockHist, day, period=10):#OR WILL USE HURST FOR THE ABOVE??-->just use the availible library?
    start_day_index = stockHist.index.get_loc(str(day))

    #average the closing price for given period past
    volumes = []
    for i in range(period):
        #print (stockHist.iloc[start_day_index-i][3])
        try:
            volumes.append(stockHist.iloc[start_day_index-i][4])
        except:
            print('problem getting stock data out, V-HURST, prob key error, not full data')
            return -1
    H, c, data = hurst.compute_Hc(volumes, kind='price', simplified=True)
    return H
"""


def volume_stdev_vs_market(stockHist, day, period=10): #sorta BETA, will reqiure storing SP here??
    pass

def price_stdev_vs_market(stockHist, day, period=10): #will reqiure storing SP here??
    pass

def pivot_point_HLO_MA(stockHist, day, period=10): #also keltner channel center
    
    start_day_index = stockHist.index.get_loc(str(day))

    total = 0
    for i in range(period):
        try:
            HLC = stockHist.iloc[start_day_index-i][1] + stockHist.iloc[start_day_index-i][2] + stockHist.iloc[start_day_index-i][3]
            total += HLC/3
        except:
            print('problem getting stock data out, HLO, prob key error, not full data')
            return -1

    return total/period

#TODO instead of constantly repeating these, should replace 
#with MA that can pick columns in arguments...


def keltner_channel_high(stockHist, day, period=10): #done simply by "typical" moving average (same as pivot)
    start_day_index = stockHist.index.get_loc(str(day))

    total = 0
    for i in range(period):
        try:
            total += stockHist.iloc[start_day_index-i][1]
        except:
            print('problem getting stock data out, KELTNER-HIGH, prob key error, not full data')
            return -1

    return total/period


def keltner_channel_low(stockHist, day, period=10): #done simply by "typical" moving average (same as pivot)
    start_day_index = stockHist.index.get_loc(str(day))

    #average the closing price for given period past
    total = 0
    for i in range(period):
        #print (stockHist.iloc[start_day_index-i][3])
        try:
            total += stockHist.iloc[start_day_index-i][2]
        except:
            print('problem getting stock data out, KELTNER-LOW, prob key error, not full data')
            return -1

    return total/period


def support(stockHist, day, period=10): #done simply by "typical" moving average (same as pivot)
    pivot = pivot_point_HLO_MA(stockHist, day, period)
    high = keltner_channel_high(stockHist, day, period)
    support = 2*pivot - high
    return support


def resistance(stockHist, day, period=10): #done simply by "typical" moving average (same as pivot)
    pivot = pivot_point_HLO_MA(stockHist, day, period)
    low = keltner_channel_low(stockHist, day, period)
    resistance = 2*pivot - low
    return resistance

def get_standard_indicators(stockHist, day):
    indicators = ['MA5', 'MA10', 'MA20', 'MA20-MA5', 'MA20-MA10','MA10-MA5',
                    'AvVolume', 'PriceStDev', 'VolStDev', 'PivotPoint',
                    'keltnerHigh', 'KeltnerLow', 'Suport', 'Resistance']

    #to adjust for weekend days
    if day.weekday() == 5 or day.weekday() == 6 or day not in stockHist.index:
        highestdate = stockHist.index[0]
        for d in stockHist.index:
            if d > highestdate and d < day:
                highestdate = d
        day = highestdate


    stockIndicts = []#FOR THE GIVEN STOCKS
    MA5 = moving_average(stockHist, day, period=5)
    MA10 = moving_average(stockHist, day, period=10)
    MA20 = moving_average(stockHist, day, period=20)
    stockIndicts.append(MA5)
    stockIndicts.append(MA10)
    stockIndicts.append(MA20)
    stockIndicts.append(MA20-MA5)
    stockIndicts.append(MA20-MA10)
    stockIndicts.append(MA10-MA5)
    stockIndicts.append(average_volume(stockHist, day))
    stockIndicts.append(price_stdev(stockHist, day))
    stockIndicts.append(volume_stdev(stockHist, day))
    #stockIndicts.append(price_hurst(stockHist, day)) #NEEDS OVER 100 values...
    #stockIndicts.append(volume_hurst(stockHist, day))
    stockIndicts.append(pivot_point_HLO_MA(stockHist, day))
    stockIndicts.append(keltner_channel_high(stockHist, day))
    stockIndicts.append(keltner_channel_low(stockHist, day))
    stockIndicts.append(support(stockHist, day))
    stockIndicts.append(resistance(stockHist, day))
    #stockIndicts.append(CBOE_volatility(stockHist, day))
    return indicators, stockIndicts


