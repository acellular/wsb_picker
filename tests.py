from datetime import date, time, datetime, timedelta
from textblob import TextBlob

import requests
import json


from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from datetime import date, time, datetime, timedelta

import StockChecker


today = date.today()
yesterday = today - timedelta(1)
print (today, yesterday)


print (date.fromtimestamp(1326244364))

d = datetime.utcnow()
print(d)
print(d.timestamp())

#1626431010
#1629877251

print(TextBlob('how are you').sentiment)

day = date.today()
print (day)
"""
VIX = StockChecker.stockHistory('^VIX', day - timedelta(60))
SKEW = StockChecker.stockHistory('^SKEW', day - timedelta(60))
print (VIX)
print (VIX.shape[0])
print (VIX.shape[0])
for i in range(1,6):
    print (VIX.iloc[VIX.shape[0]-i,0], VIX.iloc[VIX.shape[0]-i,3])
    index = VIX.index.get_loc('2021-08-26')
    print (VIX.iloc[index-1])
"""
comment = 'Idasg ddsa9fd "gdna3enteaw; g /a;lm3" ;q edfa ;g ta3tjwaet8d: ;gl afd s;d w9t dg;asgasd.'
tb = TextBlob(comment)
print (tb.words)

listtest = [[2,4,6,34,8,9,6],[432,3,2,1,24,3252,6],[5,86,6,364,23,12,43],[5,46,57,2,34,23,4]]
listmerge = listtest[:3] + listtest[7:]
print(listmerge)

listmerge2 = listtest[1] + listtest[2]
print('listmerge2',listmerge2)

listmerge3 = listtest[1].extend(listtest[2])
print('listmerge2',listmerge2)

listmerge3 = ['list', 'k'] + listtest[1] + listtest[2]
print('listmerge2',listmerge3)
