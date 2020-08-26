from Models import Rectangle


class DetectedObject:
    """
    An object that represt a detected object for a particular class
    """

    def __init__(self, box, class_id, confidence):
        self.box = box
        self.confidence = confidence
        self.class_id = class_id

    def get_rectangle(self):
        """
        :return: Rectangle representing the bounding box of a detected object
        """
        x, y, w, h = self.box
        return Rectangle(x,y,x+w,y+h)

    def is_detected_object_hard_hat(self):
        """
        :return: a bool value representing if a detetcted object is a hard hat or not, class id for har hat is 0 as
        defined in ppe.names file
        """
        return self.class_id == 0