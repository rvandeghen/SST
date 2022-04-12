"""
Script adpated from https://github.com/pytorch/vision/tree/main/references/detection
"""

import transforms as T

class DetectionPresetTrain:
    def __init__(self, hflip_prob=0.5):
        self.transforms = T.Compose([
            T.RandomPhotometricDistort(),
            T.RandomHorizontalFlip(p=hflip_prob),
            T.ToTensor(),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self):
        self.transforms = T.ToTensor()

    def __call__(self, img, target):
        return self.transforms(img, target)
