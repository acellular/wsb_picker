from datetime import date, datetime, timedelta
import time
import json

# OWN
#import LogRegress
import StockChecker


def counts_organized(csv_file, stocks_data, words, indicators):
    #stocks now instead stocks_data-->AND PULL DATE FROM STOCKS_DATA!


    with open(csv_file, "w", encoding="utf-8") as text_file: # "a" for appending
        text_file.write('Date,Stock,Percent Change (i.e. y),Count,S&P500Change,SENTIMENT-polarity,SENTIMENT-subjectivity')
        for indicator in indicators:
            text_file.write(',' + indicator)
        for word in words:
            text_file.write(',' + word)
        

        # pull stock data and add to csv
        #TODO--currently combinging stocks and date so don't have to change log regress
        for sD in stocks_data:
            #day, stock, count, change, SPchange
            text_file.write(f"\n{sD[0]},{sD[1]},{sD[2]},{sD[5]['Count']},{sD[3]}")
            #sentiment
            text_file.write(f",{sD[5]['SENTIMENT-polarity']},{sD[5]['SENTIMENT-subjectivity']}")  
            #print(sD[5])
            for sDindict in sD[4]: #indicators
                text_file.write(',' + str(sDindict))
            for word in words: #word counts
                text_file.write(',' + str(sD[5][word]))
    print ('Printed word_counts_stocks_indicators_plus_sentiment to csv')
            


def word_counts_stocks(csv_file, stocks_data, words):
    #stocks now instead stocks_data-->AND PULL DATE FROM STOCKS_DATA!


    with open(csv_file, "w", encoding="utf-8") as text_file: # "a" for appending
        text_file.write('Stock,UpDown,Count')
        for word in words:
            text_file.write(',' + word)
        

        # pull stock data and add to csv
        #TODO--currently combinging stocks and date so don't have to change log regress
        for sD in stocks_data:
            #day, stock, cat, count
            text_file.write('\n' + str(sD[0]) + str(sD[1]) + ',' + str(sD[2]) + ',' + str(sD[3])) 
            #print(sD[5])
            for word in words: #word counts
                text_file.write(',' + str(sD[5][word]))