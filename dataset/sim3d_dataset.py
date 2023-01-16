import os
import os.path as ops
import numpy as np
import cv2
import json
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F

class LaneDataset(Dataset):
    def __init__(self, args, dataset_path, json_file_path, transform=None, data_aug=False):
        self.is_testing = ('test' in json_file_path)  # 'val'
        self.num_class = args.num_class

        # define image pre-processor
        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(args.vgg_mean, args.vgg_std)
        self.data_aug = data_aug

        # dataset parameters
        self.img_path = dataset_path
        self.transform = transform
        self.h_org = args.org_h
        self.w_org = args.org_w
        self.h_crop = args.crop_y
        self.K = args.K

        # parameters related to service network
        self.h_net = args.resize_h
        self.w_net = args.resize_w
        self.H_crop = homography_crop_resize([args.org_h, args.org_w], args.crop_y, [args.resize_h, args.resize_w])

        self._label_image_path, self._label_laneline_all, \
        self._gt_cam_height_all, self._gt_cam_pitch_all, \
        self._gt_class_label_all = self.init_dataset_3D(dataset_path, json_file_path)