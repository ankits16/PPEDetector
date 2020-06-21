import cv2
import numpy as np


# safetyvest442.jpg
#
# safetyvest449
# safetyvest452 -> .4 (1 hat 2 heads)


class Rectangle:
    def intersection(self, other):
        a,b = self, other
        x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
        y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
        x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
        y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
        if x1 < x2 and y1 < y2:
            print("yes intersect")
            return True
        else:
            print("does not intersect")
            return False


    def __init__(self, x1, y1, x2, y2):
        if x1 > x2 or y1 > y2:
            raise ValueError("Coordinates are invalid")
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def getTopLeft(self):
        return (self.x1, self.y1)

    def getBottomRight(self):
        return (self.x2, self.y2)


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





net = cv2.dnn.readNetFromDarknet('/Users/ankit/Downloads/darknet-Alex/PPE/cfg/ppe_yolov3-tiny.cfg', '/Users/ankit/Downloads/darknet-Alex/ppe_yolov3-tiny_5000.weights')
classes = []
with open("/Users/ankit/Downloads/darknet-Alex/PPE/ppe.names", "r") as f:
    classes =[line.strip() for line in f.readlines()]
print("\n <<<<<<<<<<<<<<<<<<<<<all classes names \n")
print(classes)

print("\n <<<<<<<<<<<<<<<<<<<<<<< layer names \n")
layer_names = net.getLayerNames()
print(layer_names)


output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("\n <<<<<<<<<<<<<<<<<<<<<<< output layer \n")
print(output_layers)

fileInput = input("Enter file path : ")
print(fileInput)

# /Users/ankit/Desktop/PPE Dataset/PPE_84_Original/safetyvest452.jpg
img = cv2.imread(fileInput)
height, width, channels = img.shape

blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop = False)


for b in blob:
    for n,blob_img in enumerate(b):
        print("\n<<<<<<<<<<<<<<<<<<<<<< \n")
        print(blob_img)
        # cv2.imshow(str(n), blob_img)
        print("\n<<<<<<<<<<<<<<<<<<<<<< \n")


# Detection
net.setInput(blob)
outs = net.forward(output_layers)


#showing information on screen

detectedObjects = []
for out in outs:
    for detection in out:

        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.4:
            print("detection = {det} class = {id}, confidence = {c}".format(det = detection, id=class_id, c = confidence))
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            cv2.circle(img,(center_x, center_y), 10, (255,255,0),2)
            topX = int(center_x - w/2)
            topY = int(center_y - h/2)
            detectedObjects.append(DetectedObject([topX, topY, w,h],class_id, (confidence)*100))

number_objects_detected = len(detectedObjects)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
heads = [x for x in detectedObjects if x.class_id == 3]
hatOverHeadsCount = 0
for i in range(len(detectedObjects)):
    x,y,w,h = detectedObjects[i].box
    print("typs is  {type}".format(type = type(detectedObjects[i].box)))
    label = "{name} - {confidence}".format(name = str(classes[detectedObjects[i].class_id]), confidence =int( detectedObjects[i].confidence))
    color = (0, 255, 0)
    border = 1
    if detectedObjects[i].isDetectedObjectHardHat():
        print("$$$$$$$$ hard hat detected")
        hatHeadDetector = HatHeadDetetctor(detectedObjects[i], heads)
        if hatHeadDetector.checkIfHatIsPresentWithHead():
            print("%%%%%%%%%%% hat is present with head")
            hatOverHeadsCount = hatOverHeadsCount + 1
            color = (255, 153, 255)
            border = 2



    cv2.rectangle(img, detectedObjects[i].getRectangle().getTopLeft(),
                  detectedObjects[i].getRectangle().getBottomRight(), color, border)


    cv2.putText(img, label, (x,y+30),font, .75, (0,0,0), 1)




def checkIfHardHatPresent():
    print("checking")
    hats = [x for x in detectedObjects if x.class_id == 0]
    heads = [x for x in detectedObjects if x.class_id == 3]
    print("total number of hard hat = {hatCount}".format(hatCount=len(hats)))
    print("total number of heads = {headCount}".format(headCount=len(heads)))



checkIfHardHatPresent()


def checkForPPECompliance():
    compliant = False
    if len(heads) > 0 :
        if hatOverHeadsCount == len(heads):
            compliant = True
    return compliant


if checkForPPECompliance():
    print("PPE Complicant")
else:
    print("Review")

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()