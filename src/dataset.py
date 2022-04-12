"""
Script adpated from https://github.com/pytorch/vision/tree/main/references/detection
"""

import copy
import os
from PIL import Image

import torch
import torch.utils.data
import torchvision

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

import transforms as T

import json
import utils


class SNDetection(torch.utils.data.Dataset):
    
    def __init__(self, path, transforms=None):
        self.path = path
        
        self.data = json.load(open(path, 'r'))
        
        self.keys = list(self.data.keys())
        
        self.transforms = transforms
        
        self.targets = list()

        self.labels = list()
        
        for k in self.keys:
            boxes = list()
            labels = list()
            areas = list()
            iscrowds = list()
            for b in self.data[k]['bbox']:
                self.labels.append(b[4])
                boxes.append(b[:4])
                labels.append(b[4])
                a = (b[2] - b[0])*(b[3] - b[1])
                areas.append(a)
                iscrowds.append(0)
            image_id = int(k.split('/')[-1].split('.')[0])
            self.targets.append({'boxes': torch.tensor(boxes),
                                 'labels': torch.tensor(labels),
                                 'image_id': torch.tensor(image_id),
                                 'area': torch.tensor(areas),
                                 'iscrowd': torch.tensor(iscrowds)})

            if 'scores' in self.data[k].keys():
                scores = list()
                for s in self.data[k]['scores']:
                    scores.append(s)
                self.targets[-1]['scores'] = torch.tensor(scores)


        self.labels = [i for i in set(self.labels) if i > 0]
  
    def num_classes(self,):
        return len(self.labels)

    def classes(self,):
        return list(utils.CLASS_DICT.keys())

    def cls2idx(self, name):
        return utils.CLASS_DICT[name]

    def idx2cls(slef, idx):
        return utils.INVERSE_CLASS_DICT[idx]

    def __len__(self,):
        return len(self.keys)
    
    def __getitem__(self, idx):
        
        image = Image.open(self.keys[idx]).convert('RGB')
            
        targets = self.targets[idx]
        
        image, targets = self.transforms(image, targets)
        
        return image, targets

class SNUnlabeled(torch.utils.data.Dataset):
    
    def __init__(self, path):
        self.path = path
        
        self.data = json.load(open(path, 'r'))
        
        self.keys = list(self.data.keys())
        
    def __len__(self,):
        return len(self.keys)
    
    def __getitem__(self, idx):
        
        image = Image.open(self.keys[idx]).convert('RGB')
        image = torchvision.transforms.functional.to_tensor(image)
        
        return image, self.keys[idx]


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = img.shape[-2]
        img_dict['width'] = img.shape[-1]
        dataset['images'].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        iscrowd = targets['iscrowd'].tolist()
        if 'masks' in targets:
            masks = targets['masks']
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if 'keypoints' in targets:
            keypoints = targets['keypoints']
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            categories.add(labels[i])
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id
            if 'masks' in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if 'keypoints' in targets:
                ann['keypoints'] = keypoints[i]
                ann['num_keypoints'] = sum(k != 0 for k in keypoints[i][2::3])
            dataset['annotations'].append(ann)
            ann_id += 1
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    return convert_to_coco_api(dataset)

class CheckBoxes:
    def __call__(self, image, target):
        w, h = image.size

        boxes = target['boxes']
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        target['boxes'] = boxes

        return image, target

def build_dataset(root, image_set, transforms):

    ann_file = '{}_annotations.json'.format(image_set)
    ann_file = os.path.join(root, ann_file)

    t = [CheckBoxes()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    if 'unlabeled' in image_set:
        dataset = SNUnlabeled(ann_file)
    else:
        dataset = SNDetection(ann_file, transforms=transforms)

    return dataset
