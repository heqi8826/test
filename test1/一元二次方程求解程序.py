def main():
    a, b, c = eval(input('请输入所求一元二次方程的三个参数a,b,c：（a、b、c分别对应二次项系数、一次项系数、常数项）'))
    d = b**b - 4 * a * c
    if d > 0:
        print('The roots are ', format((-b + d**0.5)/(2 * a), "<.2f"), 'and', format((-b - d**0.5)/(2 * a), "<.2f"))
    if d == 0:
        print('The root is ', format(-b / (2 * a), "<.2f"))
    if d < 0:
        print('The equation has no real roots')

def game():
    main()
    count = 0
    while count >= 0:
        yourchioce = str(input('Have a try again? Y or N'))
        answer = str("Y")
        answer1 = str("y")
        if yourchioce == answer or yourchioce == answer1:
            main()
            count += 1
        else:
            print("游戏结束")
            return
game()
