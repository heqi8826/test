import math
class RegularPolygon:
    def __init__(self, n, side, x=0, y=0):
        self.__n = int(n)
        self.__side = float(side)
        self.__x = float(x)
        self.__y = float(y)

    def default_obj(self, n=3, side=1, x=0, y=0):
        self.n = n
        self.side = side
        self.x = x
        self.y = y

    def getN(self):
        return self.__n
    def getSide(self):
        return self.__side
    def getX(self):
        return self.__x
    def getY(self):
        return self.__y

    def setN(self,n):
        self.__n = int(n)
    def setSide(self,side):
        self.__side = float(side)
    def setX(self,x):
        self.__x = float(x)
    def setY(self,y):
        self.__y = float(y)

    def getPerimeter(self):
        return format(self.__n * self.__side, '7.2f')
    def getArea(self):
        return format(self.__n * self.__side * self.__n /(4 * math.tan(math.pi/self.__n)), '7.2f')
