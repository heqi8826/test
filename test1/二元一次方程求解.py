# coding = utf-8
# author = 埋头苦干的小青年
# email  = heqi88262gmail.com
# tool_version = python3.7.4
''''
ax + by = e
cx + dy = f
x = (ed - bf) / (ad - bc)
y = (a*f - e*c) / (a*d - b*c)
需求：用户输入abcdef六个参数，如果ad-bc为0，输出：the equation has no solution
'''
def main():
    a,b,c,d,e,f = eval(input('请输入六个数字参数，数字之间用英文半角逗号隔开：'))
    print("根据您的输入程序为您构建了如下二元一次方程式")
    print(str(a) + 'x' + '+', str(b)+'y' + '=', str(e))
    print(str(c) + 'x' + '+', str(d)+'y' + '=', str(f))
    print('----------')
    if a * d - b * c == 0:
        print('The equation has no solution!')
    else:
        print('x'+'=', (e*d - b*f) / (a*d - b*c))
        print('y'+'=', (a*f - e*c) / (a*d - b*c))
main()
