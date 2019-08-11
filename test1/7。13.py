class A:
    def __init__(self,i=0):
        self.__i = i
    def main():
        a = A(5)
        print a.__i
    main()