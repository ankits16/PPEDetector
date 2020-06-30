import pafy
import cv2
import numpy as np
import time
import os

from ImagePPEAuditor import ImagePPEAuditor
from PPEClasses import PPEClasses, PPEClass



class VideoPPEAuditor:
    def __init__(self):
        print("VideoPPEAuditor called")
        self.imageAuditor = ImagePPEAuditor()

    def runPPEComplianOnVideo(self):
        videoPath =  "https://www.youtube.com/watch?v=Nap8t4s0UjQ"
            # input("Enter video url:")
        self.__runDetections(videoPath)

    def __runDetections(self, videoPath):
        print("start running detection on {}".format(videoPath))
        self.predictions = []
        # vPafy = pafy.new(videoPath)
        # play = vPafy.getbestvideo(preftype="webm")
        # print(play.url)
        # cap = cv2.VideoCapture(play.url)
        # cap = cv2.VideoCapture("/Users/ankit/Downloads/PPE_3.mp4")
        cap = cv2.VideoCapture("testVideos/PPE_6.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("<<<<<< fps is {}".format(fps))
        self.frames = []
        count = 0
        success = 1
        while success:
            # Extract frame
            success, image = cap.read()
            if not count % 10:  # Write every 30th frame
                prediction = self.imageAuditor.getPPEDetection(image)
                if len(prediction.detectedHeads) > 0:
                    self.predictions.append(prediction)
                    self.frames.append(image)
                    self.imageAuditor.showDetectsOnImage()
                    # cv2.imwrite("images/frame%d.jpg" % count, image)
            count += 1


        # while (cap.isOpened()):
        #     stime = time.time()
        #     ret, frame = cap.read()
        #     if ret:
        #         prediction  = self.imageAuditor.getPPEDetection(frame)
        #         if len(prediction.detectedHeads) > 0:
        #             self.predictions.append(prediction)
        #
                # self.imageAuditor.showDetectsOnImage()
                # cv2.imshow('frame', frame)
                # print('FPS {:1f}'.format(1 / (time.time() - stime)))
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
        #     else:
        #         break
        cap.release()
        cv2.destroyAllWindows()


videoAuditor = VideoPPEAuditor()
videoAuditor.runPPEComplianOnVideo()

print("<<<<<<<<<<<<< FINAL RESULT >>>>>>>>>>>>>>>>>>\n")
i = 1
max_Head = 0
max_hat = 0
for aDetection in videoAuditor.predictions:
    if len(aDetection.detectedHats) > max_hat :
        max_hat = len(aDetection.detectedHats)

    if len(aDetection.detectedHeads) > max_Head :
        max_Head = len(aDetection.detectedHeads)

    print("frame{frame} - hats {hats} - heads {heads} - status {status}".format(frame = i, hats = len(aDetection.detectedHats), heads = len(aDetection.detectedHeads), status = aDetection.complianceStatus))
    i = i + 1
    print("\n")

resultDirectory = "images/{timeStamp}".format(timeStamp = time.time()).replace(".", "")
os.mkdir(resultDirectory)
if (max_hat) > 0 :
    ppeCompliantDetections = {}
    i = 0
    detectionsDirectory = "{}/detections".format(resultDirectory)
    os.mkdir(detectionsDirectory)
    for aDetection in videoAuditor.predictions:
        if (len(aDetection.detectedHats) == max_hat) and (len(aDetection.detectedHeads) == max_Head) :
            ppeCompliantDetections[i] = aDetection
        else:
            framePath = "{basePath}/frame{index}.jpg".format(basePath=detectionsDirectory, index=i)
            cv2.imwrite(framePath, videoAuditor.frames[i])
        i = i + 1



    if len(ppeCompliantDetections) > 0 :
        for key in ppeCompliantDetections:
            print(key)
            framePath = "{basePath}/frame{index}.jpg".format(basePath= resultDirectory, index = key)
            print(framePath)
            cv2.imwrite(framePath, videoAuditor.frames[key])
    else:
        print("\n PPE REVIEW \n")
else:
    print("\n PPE REVIEW \n")

if (max_hat > 0) and (max_hat == max_Head):
    print("\n PPE COMPLIANT \n")
else:
    print("\n PPE REVIEW \n")