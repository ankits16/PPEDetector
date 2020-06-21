
class PPEClass:
    def __init__(self, name, classId):
        self.name = name
        self.classId = classId


class PPEClasses:

    def __init__(self):
        self.rawClasses = []
        with open("/Users/ankit/Downloads/darknet-Alex/PPE/ppe.names", "r") as f:
            self.rawClasses = [line.strip() for line in f.readlines()]

        self.ppeClasses = []
        for classId, name in enumerate(self.rawClasses):
            self.ppeClasses.append(PPEClass(name, classId))

    def getAllPPEClasses(self):
        return self.ppeClasses

    def getPPEClass(self, classId):
        return  [x for x in self.ppeClasses if x.classId == classId][0]