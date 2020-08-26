
class HatHeadDetector:
    def __init__(self, hat, all_heads):
        self.hat = hat
        self.allHeads = all_heads

    def check_if_hat_is_present_over_head(self):
        """
        :return: check if a hat intersects with any of the detected human head object
        """
        is_intersected = False
        for aHead in self.allHeads:
            if aHead.get_rectangle().intersection(self.hat.get_rectangle()):
                is_intersected = True
                break
        return is_intersected
