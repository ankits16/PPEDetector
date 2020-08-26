import cv2
import numpy as np

from enum import Enum
from DetectedObject import DetectedObject
from Models import PPEClasses, PPEHardHatDetection
from HatHeadDetector import HatHeadDetector


class PPEComplianceStatus(Enum):
    Compliance = 0
    Review = 1


class ImagePPEAuditor:
    """
    Takes an image as input and checks if PPE compliance is observed in that image or not
    """
    def __initialize_image_ppe_auditor(self):

        self.net = cv2.dnn.readNetFromDarknet('yolo/ppe_yolov3-tiny.cfg', 'yolo/ppe_yolov3-tiny_5000.weights')
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.ppe_classes = PPEClasses().get_all_ppe_classes()

    def __init__(self):
        self.img = None
        self.detectedObjects = []
        self.__initialize_image_ppe_auditor()

    def run_ppe_detection(self):
        source_image = input("Enter Image path:")
        img = cv2.imread(source_image)
        self.detect_objects_in_frame(img)

    def detect_objects_in_frame(self, input_image_frame):
        """
        detect objects in an input image frame
        :param input_image_frame: target image frame on which object detection will be run
        :return: None
        """
        self.img = input_image_frame
        self.detectedObjects = []
        if not hasattr(input_image_frame, 'shape'):
            return
        height, width, channels = input_image_frame.shape
        blob = cv2.dnn.blobFromImage(input_image_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        # Detection
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # showing information on screen
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.4:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    top_x = int(center_x - w / 2)
                    top_y = int(center_y - h / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([top_x, top_y, w, h])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            top_x = box[0]
            top_y = box[1]
            w = box[2]
            h = box[3]
            self.detectedObjects.append(DetectedObject([top_x, top_y, w, h], class_ids[i], (confidences[i]) * 100))

    def get_number_of_hat_detected_on_head(self):
        """
        :return: a tuple with an array of detected hat object which are present over heads, detected head objects
        """
        hats = [x for x in self.detectedObjects if x.class_id == 0]
        heads = [x for x in self.detectedObjects if x.class_id == 3]
        hats_overheads = []
        for a_hard_hat in hats:
            hat_head_detector = HatHeadDetector(a_hard_hat, heads)
            if hat_head_detector.check_if_hat_is_present_over_head():
                hats_overheads.append(a_hard_hat)
        return hats_overheads, heads

    def get_ppe_compliance_status(self):
        """
        :return: PPEComplianceStatus
        """
        hard_hats, heads = self.get_number_of_hat_detected_on_head()
        status = PPEComplianceStatus.Review
        if len(heads) > 0:
            if len(heads) == len(hard_hats):
                status = PPEComplianceStatus.Compliance
        return status

    def get_ppe_detection(self, input_image_frame=None):
        """
        :param input_image_frame: target image frame on which object detection will be run
        :return: a PPEHardHatDetection object
        """
        self.detect_objects_in_frame(input_image_frame)
        hats, heads = self.get_number_of_hat_detected_on_head()
        return PPEHardHatDetection(hats, heads, self.get_ppe_compliance_status())

    def show_detects_on_image(self):
        """
        draw bounding boxes on image frames
        :return: None
        """
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        heads = [x for x in self.detectedObjects if x.class_id == 3]
        hat_over_heads_count = 0
        for i in range(len(self.detectedObjects)):
            x, y, w, h = self.detectedObjects[i].box
            label = "{name} - {confidence}".format(
                name=str(PPEClasses().get_ppe_class(self.detectedObjects[i].class_id).name),
                confidence=int(self.detectedObjects[i].confidence))
            color = (0, 255, 0)
            border = 1
            if self.detectedObjects[i].is_detected_object_hard_hat():
                hat_head_detector = HatHeadDetector(self.detectedObjects[i], heads)
                if hat_head_detector.check_if_hat_is_present_over_head():
                    hat_over_heads_count = hat_over_heads_count + 1
                    color = (255, 153, 255)
                    border = 2

            cv2.rectangle(self.img,
                          self.detectedObjects[i].get_rectangle().get_top_left(),
                          self.detectedObjects[i].get_rectangle().get_bottom_right(),
                          color,
                          border
                          )

            cv2.putText(self.img, label, (x, y + 30), font, .75, (0, 0, 0), 1)
