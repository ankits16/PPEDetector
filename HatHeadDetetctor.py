from  Rectangle import Rectangle
class HatHeadDetetctor:
    def __init__(self, hat, allHeads):
        self.hat = hat
        self.allHeads = allHeads

    def checkIfHatIsPresentWithHead(self):
        isIntersected = False
        for aHead in self.allHeads:
            if aHead.getRectangle().intersection(self.hat.getRectangle()):
                isIntersected =  True
                break
        return isIntersected