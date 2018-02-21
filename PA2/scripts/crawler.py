from bs4 import BeautifulSoup
import urllib

def getMonths(soup):
    items = soup.find_all('a', class_ = 'post-count-link')
    months = []
    for item in items:
        url = item.get('href')
        if len(url.split('/')[-2]) == 2:
            months.append(url)
    return months

def getArticles(month):
    r = urllib.urlopen(month).read()
    soup = BeautifulSoup(r, 'html5lib')   
    items = soup.find_all('div', dir = 'ltr')
    articles = []
    for item in items:
        text = item.text.encode('utf-8').strip()
        articles.append(text)
        print text
    return articles

if __name__ == '__main__':
    r = urllib.urlopen('http://karaiillathakadal.blogspot.in/').read()
    soup = BeautifulSoup(r, 'html5lib')
    months = getMonths(soup)
    articles = []
    for month in months:
        print month
        articles += getArticles(month)
    with open('data/blog_1.txt', 'w') as f:
        for article in articles:
            pass
            f.write(article + '\n')

