from datetime import date, datetime, timedelta
import time
import glob
import json

#local
import CommentsImporter
import StockChecker
import PullStockList
import WordCounters
import Indicators


def grab_stocklist():
    stocks = PullStockList.grab_stocklist()
    return stocks


#Preselected word list to use for counts
def pull_words():
    words = []
    with open("words.txt", encoding="utf-8") as text_file:
        words = [current_place.rstrip()
                 for current_place in text_file.readlines()]
    #print (words)
    return words


#PULL TOP N number of words in comments
def pull_words_top(comments_by_date, num_words=1000, write_to_file=False, file='comments_by_date'):
    wordCounts = {}
    for commentBD in comments_by_date:
        for cBD in commentBD[1]:
            cSplit = WordCounters.splitWords(cBD)
            for word in cSplit:
                if word != "":
                    if word in wordCounts:
                        wordCounts[word] += 1
                    else:
                        wordCounts[word] = 1

    sortedWordsByCount = sorted(wordCounts, key=wordCounts.get, reverse=True)
    sortedWordsByCount = sortedWordsByCount[:num_words]

    if write_to_file:
        with open(file + '_TOPWORDS_.txt', "w", encoding="utf-8") as f:
            for word in sortedWordsByCount :
                f.write(word + '\n')
    print ("Top words counted")
    return sortedWordsByCount


def pull_comments_by_stock(threads, stocks): #NOT TESTED!
    # THEN ADD TO FILE FOR EACH DAY
    for thread in threads:
        comments = []
        cmnt_time = datetime.utcnow().timestamp()

        for stock in stocks:
            print(stock)
            
            #AVOID OVERLOADING
            time.sleep(1)
            
            cmnts, tm = CommentsImporter.pull_comments_pushshift_bythread(stock, thread)
            print ('Time to see if increases:' + tm)
            if tm < cmnt_time: #if this comment older, just trying to get start day of thread--#inefficient
                cmnt_time = tm
            comments += cmnts
    return comments, cmnt_time


def download_comments(threads, comm_limit=1000, verbose=True):
    #comments importer
    for thread in threads:
        if verbose: print ('pulling ', thread)
        data = CommentsImporter.pull_comments_pushshift_jsononly(thread, comm_limit=comm_limit, verbose=verbose)
        filename = '.\\comments\\' + thread + '.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f)


def pull_comments_from_json(verbose=False):
    comments_by_date = []
    folder = '.\\comments\\'
    files = glob.glob(folder + '*.json')
    print ('JSON comment files imported: ', files)
    num_threads = len(files)
    for i in range(num_threads-1, -1, -1):
        comments, cmnt_time = CommentsImporter.pull_comments_from_file(files[i], verbose=verbose)
        comments_by_date.append([cmnt_time, comments])

    return comments_by_date, num_threads


def merge_returns_counts_and_indicators(comments_by_date, words, short=False, verbose=True, thread_type='What'):

    # pull stock tickers and common words
    if not short: stocks = PullStockList.grab_stocklist()
    else: stocks = PullStockList.grab_stocklist_SHORT()

    #loop through the comments
    stocks_data =[] #[day, stock, diff, counts, indicators..., sentiment and word counts...]
    stocks_histories = {} #for the dataframes from yfinance
    indicators = []

    firstday = date.today()
    lastday = date.fromtimestamp(0)

    #get date range
    for cbd in comments_by_date:
        threaddate = date.fromtimestamp(cbd[0])
        if threaddate < firstday: firstday = threaddate
        if threaddate > lastday: lastday = threaddate

    lastday = lastday+timedelta(1)#for day after last comments thread
    print (f'Stock history range needed: {firstday} to {lastday}') 

    #for adding S&P 500 change
    SP = StockChecker.stockHistory('^GSPC', firstday-timedelta(40), end=lastday+timedelta(1), verbose=verbose)

    for cbd in comments_by_date:
        day = date.fromtimestamp(cbd[0])
        print('DAY: ', day)

        #get mentions and words counts for each stock
        stock_counts = WordCounters.stock_counter_comments(cbd[1], stocks, words, verbose=verbose)

        #daily S&P 500 change (of day of thread, not day after)
        SPchange = StockChecker.stock_change(SP, day, start='sameday', verbose=verbose)

        #then find stock data!
        for i in range(len(stocks)):
            if stock_counts[i]['Count'] > 0: #counts
                #get history if needed
                if stocks[i] not in stocks_histories :
                    if verbose: print(stocks[i] + '-------------------------------------------------------')
                    try:
                        stocks_histories[stocks[i]] = StockChecker.stockHistory(stocks[i], firstday-timedelta(40), end=lastday+timedelta(1), verbose=verbose)
                    except Exception as e: #TODO-print to file?
                        print('problem downloading stock data:', e, 'STOCK: ', stocks[i])
                        continue 
                else:
                    if verbose: print('HISTORY NOT NEEDED FOR:', stocks[i])
                
                #then for this date see what percentage change in stock is
                ###This becomes the y in training###
                try:
                    diff = StockChecker.stock_change(stocks_histories[stocks[i]], day, start='open', verbose=verbose)
                except Exception as e:
                    print('problem getting stock data out:', e, 'STOCK: ', stocks[i])
                    continue         

                #indicators
                try:
                    indicators, stockIndicts = Indicators.get_standard_indicators(stocks_histories[stocks[i]], day)
                except Exception as e:
                    print('problem getting indicators:', e, 'STOCK: ', stocks[i])
                    continue
                if 'nan' in stockIndicts: #TODO-replace instead?
                    continue
                
                #Organize ready for CSV
                merge_data = [day] + [stocks[i]] + [diff] + [SPchange] + [stockIndicts] + [stock_counts[i]]
                stocks_data.append(merge_data)
        print('Stock change and indicators added.')               

    return stocks_data, indicators



if __name__ == "__main__":
    pass