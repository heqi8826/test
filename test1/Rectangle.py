class Rectangle:
    def __init__(self,width = 1 ,height = 2):
        self.width = width
        self.height = height
    def setWidth(self,width):
        self.width = width
    def setHeight(self,height):
        self.height = height
    def getWidth(self):
        return self.width
    def getHeight(self):
        return self.height
    def getPerimeter(self):
        return 2*(self.width + self.height)
    def getArea(self):
        return self.width * self.height


