import cv2
import numpy as np
from enum import Enum

from DetectedObject import DetectedObject
from PPEClasses import PPEClasses
from HatHeadDetetctor import HatHeadDetetctor


class PPEComplianceStatus(Enum):
    Compliance = 0
    Review = 1



class PPEDetection:
    def __init__(self, detectedHats, detectedHeads, complianceStatus):
        self.detectedHats = detectedHats
        self.detectedHeads = detectedHeads
        self.complianceStatus = complianceStatus


class ImagePPEAuditor:

    def __initializeImagePPEAuditor(self):
        print("initializeCalled")
        self.net = cv2.dnn.readNetFromDarknet('/Users/ankit/Downloads/darknet-Alex/PPE/cfg/ppe_yolov3-tiny.cfg',
                                         '/Users/ankit/Downloads/darknet-Alex/ppe_yolov3-tiny_5000.weights')
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.ppeClasses = PPEClasses().getAllPPEClasses()


    def __init__(self):
        self.__initializeImagePPEAuditor()

    def runPPEDetection(self):
        sourceImage = input("Enter Image path:")
        img = cv2.imread(sourceImage)
        self.detectObjectsInFrame(img)

    def detectObjectsInFrame(self, img):
        self.img = img
        self.detectedObjects = []
        if not hasattr(img, 'shape'):
            return
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        # Detection
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # showing information on screen
        self.detectedObjects = []
        for out in outs:
            for detection in out:

                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.4:
                    print("class = {id}, confidence = {c}".format(id=PPEClasses().getPPEClass(class_id).name, c=confidence))
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # cv2.circle(img, (center_x, center_y), 10, (255, 255, 0), 2)

                    topX = int(center_x - w / 2)
                    topY = int(center_y - h / 2)
                    self.detectedObjects.append(DetectedObject([topX, topY, w, h], class_id, (confidence) * 100))

        number_objects_detected = len(self.detectedObjects)
        print("\n<<<<<<<< detected objects = {num}\n".format(num=number_objects_detected))

    def getNumberOfHatDetectedOnHead(self):
        hats = [x for x in self.detectedObjects if x.class_id == 0]
        heads = [x for x in self.detectedObjects if x.class_id == 3]
        hatsOverheads = []
        for aHardHat in hats:
            hatHeadDetector = HatHeadDetetctor(aHardHat, heads)
            if hatHeadDetector.checkIfHatIsPresentWithHead():
                hatsOverheads.append(aHardHat)
        return (hatsOverheads, heads)

    def getPPEComplianceStatus(self):
        hardHats, heads = self.getNumberOfHatDetectedOnHead()
        status = PPEComplianceStatus.Review
        if len(heads) > 0:
            if len(heads) == len(hardHats):
                status = PPEComplianceStatus.Compliance
        return status

    def getPPEDetection(self, img = None):
        # if img == None:
        #     self.runPPEDetection()
        # else:
        #     self.detectObjectsInFrame(img)

        self.detectObjectsInFrame(img)
        hats, heads = self.getNumberOfHatDetectedOnHead()
        return PPEDetection(hats,heads,self.getPPEComplianceStatus())

    def showDetectsOnImage(self):

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        heads = [x for x in self.detectedObjects if x.class_id == 3]
        hatOverHeadsCount = 0
        for i in range(len(self.detectedObjects)):
            x, y, w, h = self.detectedObjects[i].box
            print("typs is  {type}".format(type=type(self.detectedObjects[i].box)))
            label = "{name} - {confidence}".format(name=str(PPEClasses().getPPEClass(self.detectedObjects[i].class_id).name),
                                                   confidence=int(self.detectedObjects[i].confidence))
            color = (0, 255, 0)
            border = 1
            if self.detectedObjects[i].isDetectedObjectHardHat():
                print("$$$$$$$$ hard hat detected")
                hatHeadDetector = HatHeadDetetctor(self.detectedObjects[i], heads)
                if hatHeadDetector.checkIfHatIsPresentWithHead():
                    print("%%%%%%%%%%% hat is present with head")
                    hatOverHeadsCount = hatOverHeadsCount + 1
                    color = (255, 153, 255)
                    border = 2

            cv2.rectangle(self.img, self.detectedObjects[i].getRectangle().getTopLeft(),
                          self.detectedObjects[i].getRectangle().getBottomRight(), color, border)

            cv2.putText(self.img, label, (x, y + 30), font, .75, (0, 0, 0), 1)
            # cv2.imshow("Image", self.img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


# auditor = ImagePPEAuditor()
# detection = auditor.getPPEDetection()
# print(detection.complianceStatus)
# auditor.showDetectsOnImage()
# auditor.runPPEDetection()
# hardHats, heads = auditor.getNumberOfHatDetectedOnHead()
#
# print("hats = {}, heads = {}".format(len(hardHats), len(heads)))
#
# print(auditor.getPPEComplianceStatus())