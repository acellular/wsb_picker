from selenium import webdriver

import time



# setup webdriver
def grab_html(url):
    driver = webdriver.Chrome(executable_path=r'c:\chromedriver.exe')
    driver.get(url)
    return driver


# get links to threads via class type
def grab_link(driver, startswith, scrolls):
    
    dailylinks = []

    for i in range(scrolls):
        
        links = driver.find_elements_by_xpath('//h3[@class="_eYtD2XCVieq6emjKBH3m"]')
        for a in links:
            if a.text.startswith(startswith):
                link = a.find_element_by_xpath('../..').get_attribute('href')
                dailylinks.append(link.split('/')[-3])
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

    print('UNCUT LINKS: ', dailylinks)
    dailylinks = list(dict.fromkeys(dailylinks)) #REMOVE DUPLICATES

    #print (dailylinks)
    return dailylinks


# change skip to get todays data
def pull_threads(thread_name_starts, sub_link, num_days_back, skip_first_days=1):

    # pull recent comments from reddit daily threads
    driver = grab_html(sub_link)   
    thread_links = grab_link(driver, thread_name_starts, 10)
    driver.close()
   
    #skip threads (mostly to skip current day's incomplete thread)
    thread_links = thread_links[skip_first_days:]

    #cut threads to number of days requested
    if len(thread_links) > num_days_back:
        thread_links = thread_links[:num_days_back]


    print ('Thread links: ', thread_links)
    return thread_links


#TEST
if __name__ == "__main__":
    url = 'https://www.reddit.com/r/wallstreetbets/search/?q=flair%3A%22Daily%20Discussion%22&restrict_sr=1&sort=new'
    thread = pull_threads('What', url, 1)
    print (thread)