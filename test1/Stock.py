class Stock:
    def __init__(self,symbol,name,previousClosingPrice,currentPrice):
        self.__symbol = str(symbol)
        self.__name = str(name)
        self.__previousClosingPrice = float(previousClosingPrice)
        self.__currentPrice = float(currentPrice)
    # stock = stock(symbol,name,previousClosingPrice,currentPrice)
    def getName(self):
        return self.__name
    def getSymbol(self):
        return self.__symbol
    def getPreviousClosingPrice(self):
        return self.__previousClosingPrice
    def getCurrentPrice(self):
        return self.__currentPrice
    def setPreviousClosingPrice(self):
        self.__previousClosingPrice = previousClosingPrice
    def setCurrentPrice(self):
        self.__currentPrice = currentPrice
    def getChagePercent(self):
        return  format((self.__currentPrice - self.__previousClosingPrice) * 100/ self.__previousClosingPrice, '5.2f') + "%"