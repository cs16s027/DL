from bs4 import BeautifulSoup
import urllib
import justext

def getURLs(soup):
    urls = []
    items = soup.find_all('tr')
    for item in items:
        items_ = item.find_all('td')
        for item_ in items_:
            if item_.a == None:
                print item_.text.encode('utf8')
            else:
                url = item_.a.get('href')
                print url
        continue
        if item.a == None:
            continue
        else:
            url_ = item.a.get('href')
        if 'utf8' not in url_:
            continue
        url = '{}{}'.format(base_url, url_)
        if 'html' not in url:
            continue
        urls.append(url)
    return urls

def getContent(url):
    r = urllib.urlopen(url).read()
    soup = BeautifulSoup(r, 'html5lib')
    return soup.get_text().encode('utf8')

if __name__ == '__main__':
    base_url = 'http://www.projectmadurai.org'
    r = urllib.urlopen('http://www.projectmadurai.org/pmworks.html').read()
    soup = BeautifulSoup(r, 'html5lib')
    urls = getURLs(soup)

