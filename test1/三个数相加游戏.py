import random
def main():
    a = random.randint(0, 9)
    b = random.randint(0, 9)
    c = random.randint(0, 9)

    result = eval(input(str(a)+'+'+str(b)+'+'+str(c)+'='))
    if a + b + c == result:
        print('Congratulations')
    else:
        print('wrong!')
main()
