from Rectangle import Rectangle

def main():
    juxing1 = Rectangle()
    juxing1.setWidth(4)
    juxing1.setHeight(40)
    print("the width is", juxing1.getWidth())
    print("the height is", juxing1.getHeight())
    print("the Perimeter is", juxing1.getPerimeter())
    print("the Area is", juxing1.getArea())
main()

def main1():
    juxing2 = Rectangle()
    juxing2.setWidth(3.5)
    juxing2.setHeight(35.7)
    print("the width is", juxing2.getWidth())
    print("the height is", juxing2.getHeight())
    print("the Perimeter is", juxing2.getPerimeter())
    print("the Area is", juxing2.getArea())
main1()

print(3.5 * 35.7)