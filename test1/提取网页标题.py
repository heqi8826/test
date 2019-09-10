
# author = heqi8826
# email  = heqi8826@gmail.com
# language_v = python3.7.4
from urllib import request
import re
def main():
    myurl = input('采集目标网址为，必须输入http(s)://协议：')
    data = request.urlopen(myurl).read().decode('utf-8')
    mycontent = re.findall('<content>(.+?)</content>', data)
    print(mycontent)
main()

'''
data = request.urlopen(myurl).read().decode('utf-8') 没有进行读取read操作和 编码转化decode 导致报错
'''