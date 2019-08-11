# coding = utf-8
birthspeed = 1/7
diespeed = 1/13
movespeed = 1/45
nowsum = 3120324986
yearseconds = 365*24*60*60
class newsum:
    def getNewsum(self,n):
        self.n = n
        yearseconds = 365 * 24 * 60 * 60
        newsum = (yearseconds * 1 // 45 + yearseconds * 1 // 7 - yearseconds * 1 // 13) * n +nowsum
        return newsum
def main():
    c = newsum()


    for i in range(5):
        n = 1
        print 'the first year is ',c.getNewsum(n)
        n += 1

main()