from textblob import TextBlob


# split the words of a comment (or any other text)
def splitWords(text):
    text = text.replace(',', '')
    text = text.replace(';', '')
    text = text.replace('.', '')
    text = text.replace('$', '')
    text = text.replace('\n\n', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    words = text.split(' ')
    return words


def stock_counter_comments(comments, stocks, words, verbose=False):
    
    #initialize dict for holding stock mentions count and words count
    stocks_counter = [{}] * len(stocks)
    for i in range(len(stocks)):
        stocks_counter[i] ={'Count':0, 'SENTIMENT-polarity':0, 'SENTIMENT-subjectivity':0} | {word: 0 for word in words}

    #to search for stock mentions
    comments_split = [splitWords(comment) for comment in comments] #[[]]

    #for every stock, see if mentioned in any comment, and count words if so
    for i in range(len(stocks)): #STOCKS
        addedPol = 0
        addedSub = 0
        for f in range(len(comments)): #COMMENTS
            
            #is the stock mentioned?
            commentSplit = comments_split[f] 
            found = False
            if stocks[i] in commentSplit:
                found = True
                stocks_counter[i]['Count'] += 1

            #if so, add to the count of stock and related words
            if found: #print("found", stocks[i])
                for word in words: #WORDS IN COMMENT
                    for g in range(len(commentSplit)): 
                        if word == commentSplit[g]:
                            stocks_counter[i][word] += 1
                            break #TODO--WOULD WORK BETTER TO COUNT ALL??
                #sentiment
                commentBlob = TextBlob(comments[f])
                addedPol += commentBlob.sentiment.polarity
                addedSub += commentBlob.sentiment.subjectivity

        if stocks_counter[i]['Count'] > 0:
            stocks_counter[i]['SENTIMENT-polarity'] = addedPol/stocks_counter[i]['Count'] 
            stocks_counter[i]['SENTIMENT-subjectivity'] = addedSub/stocks_counter[i]['Count']
            if verbose: print(f"{stocks[i]} mentioned {stocks_counter[i]['Count']} times")
    
    print('Stocks counted')        
    return stocks_counter