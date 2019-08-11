from Stock import Stock
def main():
    stock1 = Stock('INTC', 'Intel Corporation', 20.5, 20.35)
    print(stock1.getName(), '公司的股票符号是：', stock1.getSymbol())
    print(stock1.getName(), '前一天的股票收盘价格是：', stock1.getPreviousClosingPrice())
    print(stock1.getName(), '当前的股票收盘价格是：', stock1.getCurrentPrice())
    print(stock1.getName(), '今天收盘价与前一天收盘价变动了', stock1.getChagePercent())
main()