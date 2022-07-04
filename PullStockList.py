


def grab_stocklist():
    blacklist_words = [
        "YOLO", "TOS", "CEO", "CFO", "CTO", "DD", "BTFD", "WSB", "OK", "RH",
        "KYS", "FD", "TYS", "US", "USA", "IT", "ATH", "RIP", "BMW", "GDP",
        "OTM", "ATM", "ITM", "IMO", "LOL", "DOJ", "BE", "PR", "PC", "ICE",
        "TYS", "ISIS", "PRAY", "PT", "FBI", "SEC", "GOD", "NOT", "POS", "COD",
        "AYYMD", "FOMO", "TL;DR", "EDIT", "STILL", "LGMA", "WTF", "RAW", "PM",
        "LMAO", "LMFAO", "ROFL", "EZ", "RED", "BEZOS", "TICK", "IS", "DOW",
        "AM", "PM", "LPT", "GOAT", "FL", "CA", "IL", "PDFUA", "MACD", "HQ",
        "OP", "DJIA", "PS", "AH", "TL", "DR", "JAN", "FEB", "JUL", "AUG",
        "SEP", "SEPT", "OCT", "NOV", "DEC", "FDA", "IV", "ER", "IPO", "RISE",
        "IPA", "URL", "MILF", "BUT", "SSN", "FIFA", "USD", "CPU", "AT",
        "GG", "ELON", "ALL", "GO", "LOVE", "PUMP", "ON", 'HUGE',
        "BIG", "BRO", "GF", "BJ", "THC", "WELL", "HE", "FOR", "HES", "DOW", 'HOME',
        "LIVE", "OPEN", "RIOT", "TEAM", "TALK", "VERY", "TRUE", "LMAO", "NICE", 'CDC',
        "REAL", "TELL", #IT?, IQ? EAT? DM? BASE? BBQ? OG? FAST? OLD? ROCK? FLY? STEM? SAVE?
        'HAS', 'AN', 'CAN', 'HAS', 'COST', 'BIG', 'RIDE', 'DOW', 'ETH', 'IRS', 'NSA', 'NYC', 'ONE',
        'OUT', 'PAY', 'RE', 'SHOP', 'TV', 'TECH', 'TIL', 'AGO', 'AC', 'LIFE', 'MF', 'BO', 
        'BUD', 'FACT', 'LEAP','NEW', 'MOVE', 'PLAY', 'RUN', 'WOOF', 'CASH', 'ARE',
        "LGBT", "FANG", "TURN", "ANY", "NOW", "PM", "WAT", "HOPE", "ON", "NEXT", "GOLD", 
        "COIN", "GOOD", "CAP", "OR", "SAFE", "YOU", "ALOT", 'MAN', "UP", "FREE", "ME", "NEXT",
        'EVER', 'FAT', "HUGE", 'WOW', 'LOW', 'FIVE', 'BEST', 'UK','CBD', 'PLAN', 'CIA', 'SJW',
        'FLY', 'WORK', 'AA', 'GO', 'AI', 'AIR'
    ]
    #TODO->different cut lists for predictions and training
    #print (blacklist_words)
    #MAY WANT DIFFERENT LIST FOR PULLING PREDICTIONS-->much less strict

    with open('combinedNYSEandNASDAQ.txt', 'r') as w:
        w.readline()
        stocks = w.readlines()
        #print(stocks)
        stocks_list = []
        for a in stocks:
            stock = a.split('\t')[0]
            # don't really need blacklist untill pull from comments
            if len(stock) > 1 and '-' not in stock and '.' not in stock and not stock in blacklist_words:
                stocks_list.append(stock)

        #print(stocks_list)

        return stocks_list

def grab_stocklist_SHORT():
    stocks = grab_stocklist()
    stocks = stocks[:200]
    #print(stocks)
    return stocks

#stocks = grab_stocklist()
    #with open('stocklist.txt', 'w') as w:
    #    w.writelines(stocks)