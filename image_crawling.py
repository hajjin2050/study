from urllib.request import urlopen
from bs4 import BeautifulSoup

# html = urlopen("http://www.naver.com")  
# bsObject = BeautifulSoup(html, "html.parser") 
# print(bsObject)

with urlopen('http://en.wikipedia.org/wiki/Main_Page')as response:
    soup = BeautifulSoup(response, 'html.parser')
    for anchor in soup.find_all('a'):
        print(anchor.get('href', '/'))