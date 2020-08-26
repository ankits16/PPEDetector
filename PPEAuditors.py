import cv2
import time
import os

from Models import VideoPPEAuditResult


class HardHatAuditor:
    """
    Audits the hard hats
    """

    def audit_hard_hats(self, predictions, captured_frames):
        """
        Checks the maximum number of hard hats and human heads in a list of predictions, if maximum number of hats are
        greater than zero and equal to maximum number of heads in a scene, the frame is PPE compliance
        :param predictions: predictions returned by neural net
        :param captured_frames: frames from video or images
        :return: PPE Audit result
        """

        max_head = 0
        max_hat = 0
        for aDetection in predictions:
            if len(aDetection.detectedHats) > max_hat:
                max_hat = len(aDetection.detectedHats)

            if len(aDetection.detectedHeads) > max_head:
                max_head = len(aDetection.detectedHeads)

        result_directory = "images/{timeStamp}".format(timeStamp=time.time()).replace(".", "")
        os.mkdir(result_directory)
        if max_hat > 0:
            ppe_compliant_detections = {}
            counter = 0
            detections_directory = "{}/detections".format(result_directory)
            os.mkdir(detections_directory)
            for aDetection in predictions:
                if (len(aDetection.detectedHats) == max_hat) and (len(aDetection.detectedHeads) == max_head):
                    ppe_compliant_detections[counter] = aDetection
                else:
                    pass
                counter = counter + 1
            evidence_frame_paths = []
            if len(ppe_compliant_detections) > 0:
                for key in ppe_compliant_detections:
                    print(key)
                    frame_path = "{basePath}/frame{index}.jpg".format(basePath=detections_directory, index=key)
                    evidence_frame_paths.append(frame_path)
                    print(frame_path)
                    cv2.imwrite(frame_path, captured_frames[key])
            else:
                return VideoPPEAuditResult(False, None)
        else:
            return VideoPPEAuditResult(False, None)

        if (max_hat > 0) and (max_hat == max_head):
            return VideoPPEAuditResult(True, evidence_frame_paths)
        else:
            return VideoPPEAuditResult(False, None)

