from Rectangle import Rectangle

class DetectedObject:
    def __init__(self, box, class_id, confidence):
        self.box = box
        self.confidence = confidence
        self.class_id = class_id

    def getRectangle(self):
        x, y, w, h = self.box
        return Rectangle(x,y,x+w,y+h)

    def isDetectedObjectHardHat(self):
        return  self.class_id == 0