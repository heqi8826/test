from urllib import request
from bs4 import BeautifulSoup


def main():
    myurl = input('输入要采集的网页网址，必须输入http(s)协议：')
    data = request.urlopen(myurl)
    print(BeautifulSoup(data, 'html.parser').content)


main()
