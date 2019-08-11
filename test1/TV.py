# coding=utf-8
class TV:
    def __init__(self):
        self.channel = 1
        self.volumeLevel = 1
        self.on = False
    def turnOn(self):
        self.on = True
    def turnOff(self):
        self.on = False
    def getChannel(self):
        return channel
    def setChannel(self):
        if self.on == True and 1 <= self.channel <= 120:
            self.channel = channel
    def getVolume(self):
        return self.volumeLevel
    def setVolume(self):
        if self.on == True and 1 <= self.volumeLevel <= 7:
            self.volumeLevel = volumeLevel
    def channelUp(self):
        if self.on == True and 1 <= self.channel < 120:
            self.channel = channel + 1
        else:
            print '没有更多的频道了'
    def channelDown(self):
        if self.on == True and 1 < self.channel <= 120:
            self.channel = channel - 1
        else:
            print 'no much less channel'
    def volumeUp(self):
        if self.on == True and 1 <= self.volumeLevel < 7:
            self.volumeLevel += 1
        else:
            print 'vol has already been highest'
    def volumeDown(self):
        if self.on == True and 1 < self.volumeLevel <= 7:
            self.volumeLevel -= 1
        else:
            print 'vol has already been lowest'



