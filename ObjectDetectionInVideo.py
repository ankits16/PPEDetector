import pafy
import cv2
import numpy as np
import time
from opencvExperiment import Rectangle

net = cv2.dnn.readNetFromDarknet('/Users/ankit/Downloads/darknet-Alex/PPE/cfg/ppe_yolov3-tiny.cfg', '/Users/ankit/Downloads/darknet-Alex/ppe_yolov3-tiny_5000.weights')
classes = []
with open("/Users/ankit/Downloads/darknet-Alex/PPE/ppe.names", "r") as f:
    classes =[line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()



output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("\n <<<<<<<<<<<<<<<<<<<<<<< output layer \n")
print(output_layers)

# url = "https://www.youtube.com/watch?v=q8NDe_fqwWU"
# url = "https://www.youtube.com/watch?v=B15aSPGvHS8"
# url = "https://www.youtube.com/watch?v=cDY2Imadffk"
# vPafy = pafy.new(url)
# play = vPafy.getbestvideo(preftype="webm")
# print(play.url)
#
# # cap = cv2.VideoCapture("/Users/ankit/Downloads/darknet-Alex/PPE/unseen/demoVideo.mp4")
# cap = cv2.VideoCapture(play.url)
cap = cv2.VideoCapture("/Users/ankit/Downloads/PPE_6.mp4")
colors=[tuple(255 * np.random.rand(3)) for i in range(5)]

def detectObjectsInFrame(img):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Detection
    net.setInput(blob)
    outs = net.forward(output_layers)

    # showing information on screen
    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                print("class = {id}, confidence = {c}".format(id=class_id, c=confidence))
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                cv2.circle(img, (center_x, center_y), 10, (255, 255, 0), 2)

                topX = int(center_x - w / 2)
                topY = int(center_y - h / 2)
                boxes.append([topX, topY, w, h])
                confidences.append((confidence) * 100)
                class_ids.append(class_id)

    number_objects_detected = len(boxes)
    print("\n<<<<<<<< detected objects = {num}\n".format(num = number_objects_detected))

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = "{name} - {confidence}".format(name=str(classes[class_ids[i]]), confidence=int(confidences[i]))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y + 30), font, .75, (0, 0, 0), 2)

while(cap.isOpened()):
    stime = time.time()
    ret, frame = cap.read()
    if ret:
        # for color, result in zip(colors, results):
        #     tl = (result['topleft']['x'], result['topleft']['y'])
        #     br = (result['bottomright']['x'], result['bottomright']['y'])
        #     label = result['label']
        #     frame = cv2.rectangle(frame, tl, br, color, 7)
        #     frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        detectObjectsInFrame(frame)
        cv2.imshow('frame', frame)
        print('FPS {:1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

