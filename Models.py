"""
All the models used in PPE detection
"""

class PPEClass:
    """
    Represents a PPE class defined in ppe.names
    """
    def __init__(self, name, class_id):
        self.name = name
        self.classId = class_id


class PPEClasses:
    """
    represents a collection of all the PPE classes defined in ppe.names
    """
    def __init__(self):
        self.rawClasses = []
        with open("yolo/ppe.names", "r") as f:
            self.rawClasses = [line.strip() for line in f.readlines()]

        self.ppeClasses = []
        for classId, name in enumerate(self.rawClasses):
            self.ppeClasses.append(PPEClass(name, classId))

    def get_all_ppe_classes(self):
        """
        :return: an array of all the PPE classes defined in ppe.names
        """
        return self.ppeClasses

    def get_ppe_class(self, class_id):
        """
        :param class_id: identifier for a predefined object class in ppe.names
        :return: a PPE class
        """
        return [x for x in self.ppeClasses if x.classId == class_id][0]


class VideoPPEAuditResult:
    def __init__(self, is_ppe_compliant, evidence_frame_path):
        """
        :param is_ppe_compliant: a bool flag if PPE compliance is there or not
        :param evidence_frame_path: path of evidence frame, it will be null if PPE compliance is not followed
        """
        self.is_ppe_compliant = is_ppe_compliant
        self.evidence_frame_Path = evidence_frame_path


class Rectangle:
    """
    Represents a geometrical rectangle
    """
    def intersection(self, other_rectangle):
        """
        :param other_rectangle: target rectangle
        :return: a bool value representing if a current rinstance of rectangle intersects with anothe rectangle or not
        """
        a,b = self, other_rectangle
        x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
        y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
        x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
        y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
        if x1 < x2 and y1 < y2:
            return True
        else:
            return False

    def __init__(self, x1, y1, x2, y2):
        if x1 > x2 or y1 > y2:
            raise ValueError("Coordinates are invalid")
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def get_top_left(self):
        """
        :return:  top left coordinates of a rectangle
        """
        return self.x1, self.y1

    def get_bottom_right(self):
        """
        :return: botom right coordinates of a rectangle
        """
        return self.x2, self.y2


class PPEHardHatDetection:
    """
    A detection object encapsulating detected hats , human heads and compliance status
    """
    def __init__(self, detected_hats, detected_heads, compliance_status):
        self.detectedHats = detected_hats
        self.detectedHeads = detected_heads
        self.complianceStatus = compliance_status
