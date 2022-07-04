
import requests
import json
import praw

import ThreadsFinder

def json_extract(obj, key):
        """Recursively fetch values from nested JSON."""
        arr = []

        def extract(obj, arr, key):
            """Recursively search for values of key in JSON tree."""
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        extract(v, arr, key)
                    elif k == key:
                        arr.append(v)
            elif isinstance(obj, list):
                for item in obj:
                    extract(item, arr, key)
            return arr

        values = extract(obj, arr, key)
        return values

#FOR RECENT--highly throttled
def pull_commentsPRAW(link):
    comments = []
    reddit = praw.Reddit(client_id="-ZMBTknFI5BhQ1smjJkOgw",
                         client_secret="ywf-OmAoiMTprA3sfm9hC3gtFaL0YQ",
                         user_agent="crawler_test")
    submission = reddit.submission(id=link)

    submission.comments.replace_more(limit=3000, threshold=5)#threshold
    print('passed')
    for comment in submission.comments.list():
        comments.append(comment.body)
    time = submission.created

    print(comments)
    print(len(comments)) 
    print (time)

    return comments, time

#download comments as json
def pull_comments_pushshift_jsononly(thread_link, comm_limit=1000, verbose=True):
    html = requests.get(
        f'https://api.pushshift.io/reddit/comment/search/?link_id={thread_link}&limit={comm_limit}')

    #print(html)
    comment_data = html.json()
    if verbose: print (comment_data)

    #if not comments returned
    if len(comment_data['data']) > 0:
        return comment_data
    else:
        print('NO COMMENTS YET')
        return None


#ALT using stored comments
def pull_comments_from_file(file, verbose=True):
    with open(file) as f:
        comment_data = json.load(f)
    comments = json_extract(comment_data, 'body')
    time = 0

    #if not comments returned
    if len(comment_data['data']) > 0:
        #LATER COULD ADD OTHER INFO LIKE FLAIRS ETC???
        utclist = json_extract(comment_data, 'created_utc')
        time = utclist[len(utclist)-1]
        print(f'Number of comments imported: {len(comments)}, for thread starting at time: {time}') 
        if verbose:
            print(comments)
            print(time)

    return comments, time


#TESTING
#pull_comments_pushshift_bythreadonly('pa7km0')

#to pull comments and save to folder-->later turn into method run by datapuller only if needed?
#-->SAVE AS THREAD CODE so can check easily
if __name__ == "__main__":

    url = 'https://www.reddit.com/r/wallstreetbets/search/?q=flair%3A%22Daily%20Discussion%22&restrict_sr=1&sort=new'
    thread_type = 'What'
    num_days_back = 30

    #PULL THREAD CODES
    #threads = ['ps2twg'] #TEMP! , 'pa7km0' ['pa7km0', 'p9jpeq', 'p7mehd', 'p6yxxc']

    #-->skip_first_days=0 for same day thread (for prediction usually)
    threads = ThreadsFinder.pull_threads(thread_type, url, num_days_back, skip_first_days=2)

    for thread in threads:
        print ('pulling ', thread)
        data = pull_comments_pushshift_jsononly(thread)
        filename = '.\\comments\\'+thread + '.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        