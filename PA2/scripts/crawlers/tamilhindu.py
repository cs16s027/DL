from bs4 import BeautifulSoup
import urllib
import justext

if __name__ == '__main__':
    base_url = 'http://tamil.thehindu.com/archive/'
    r = urllib.urlopen('http://tamil.thehindu.com/archive/').read()
    soup = BeautifulSoup(r, 'html5lib')
    print soup
