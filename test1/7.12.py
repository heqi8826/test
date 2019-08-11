class Count:
    def __init__(self,count = 0):
        self.count = count
def main():
    c = Count()
    times = 1
    increment(c, times)
    print 'count is',c.count
    print 'times is',times
def increment(c,times):
    c = Count(5)
    times = 3
main()