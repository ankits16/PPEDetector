import cv2
import VideoPPEAuditorExceptions

from PPEAuditors import HardHatAuditor
from Models import VideoPPEAuditResult
from os import path
from ImagePPEAuditor import ImagePPEAuditor


class VideoPPEAuditor:
    """VideoPPEAuditor analyzes a video
    and does following things:
    1) Checks if the PPE compliance  is practiced in the video
    2) Extracts a frame from the video
    """

    def __init__(self):
        print("VideoPPEAuditor called")
        self.imageAuditor = ImagePPEAuditor()

    def run_ppe_compliance_on_video(self, video_path):
        """
        runs PPE compliance on a video
        :param video_path: path of the video which needs to be ananylzed
        :return:
        """

        try:
            return self.__run_detections(video_path)
        except VideoPPEAuditorExceptions.VideoFileNotFoundError as e:
            print("<<<<<<<<<< error ")
            return VideoPPEAuditResult(False, None)

    def __run_detections(self, video_path):
        print("start running detection on {}".format(video_path))
        self.predictions = []

        if path.exists(video_path):
            print("<<<<<< file exists")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print("<<<<<< fps is {}".format(fps))
            self.frames = []
            count = 0
            success = 1
            while success:
                success, image = cap.read()
                if not count % 10:
                    prediction = self.imageAuditor.get_ppe_detection(image)
                    if len(prediction.detectedHeads) > 0:
                        self.predictions.append(prediction)
                        self.frames.append(image)
                        self.imageAuditor.show_detects_on_image()
                count += 1
            cap.release()
            cv2.destroyAllWindows()
            return self.__run_inference_on_detections(self.predictions)
        else:
            raise VideoPPEAuditorExceptions.VideoFileNotFoundError

    def __run_inference_on_detections(self, predictions):
        if not predictions:
            return VideoPPEAuditResult(False, None)
        else:
            return HardHatAuditor().audit_hard_hats(predictions, self.frames)


# HOW TO USE:-


videoAuditor = VideoPPEAuditor()
result = videoAuditor.run_ppe_compliance_on_video("testVideos/PPE_1.mp4")

print("<<<<<<<<<<<<< FINAL RESULT >>>>>>>>>>>>>>>>>>\n")
if result.is_ppe_compliant:
    print("Compliant")
else:
    print("Review")
