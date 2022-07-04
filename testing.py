from datetime import date
import wsbCrawler as wsb
import ThreadsFinder
import PrintToCSV

thread_type = 'What'
#DOWNLOAD NEW COMMENTS FROM LATEST THREADS AND SAVE (if wanted)
#NOTE: WILL TRY TO OPEN A CROME BROWSER WINDOW
#TODO--replace with API method?
"""
num_days_back = 2
url = 'https://www.reddit.com/r/wallstreetbets/search/?q=flair%3A%22Daily%20Discussion%22&restrict_sr=1&sort=new'
threads = ThreadsFinder.pull_threads(thread_type, url, num_days_back, skip_first_days=1)
wsb.download_comments(threads, comm_limit=2000)
"""


#LOAD IN COMMENTS
comments_by_date, num_threads = wsb.pull_comments_from_json(verbose=False)

#WORDS TO COUNT
words = wsb.pull_words_top(comments_by_date, num_words=1000)
#print (words)



#get daily returns and indicators on mentioned stocks
stocks_data, indicators = wsb.merge_returns_counts_and_indicators(comments_by_date, words, short=False, verbose=True)


#NAME THE CSV FILE
lastday = date.fromtimestamp(comments_by_date[0][0])
firstday = date.fromtimestamp(comments_by_date[len(comments_by_date)-1][0])
csv_file = f'counts_{thread_type}--from{firstday}_to_{lastday}.csv'

PrintToCSV.counts_organized(csv_file, stocks_data, words, indicators)