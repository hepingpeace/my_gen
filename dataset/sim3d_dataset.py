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

        def __len__(self):
            return len(self._label_image_path)

        def __getitem__(self, idx):
            img_name = self._label_image_path[idx]
            cam_pitch = self._gt_cam_pitch_all[idx]
            cam_height = self._gt_cam_height_all[idx]

            # 图片位置
            # 将图片转为RGB格式
            with open(img_name, 'rb') as f:
                image = (Image.open(f).convert('RGB'))

            # image preprocess with crop and resize
            # 图像预处理裁剪和调整大小
            image = F.crop(image, self.h_crop, 0, self.h_org - self.h_crop, self.w_org)
            image = F.resize(image, size=(self.h_net, self.w_net),
                             interpolation=transforms.InterpolationMode.BILINEAR)  # interpolation=Image.BILINEAR
            if self.data_aug:
                img_rot, aug_mat = data_aug_rotate(image)
                image = Image.fromarray(img_rot)
            image = self.totensor(image).float()  # 将张量（tensor）转换为 float 浮点数
            image = self.normalize(image)  # 将图像进行归一化

            # prepare binary segmentation label map
            # 准备二进制分割标签图
            label_map = np.zeros((self.h_net, self.w_net), dtype=np.int8)  # label—map 于照片大小一样
            gt_lanes = self._label_laneline_all[idx]
            gt_labels = self._gt_class_label_all[idx]
            for i, lane in enumerate(gt_lanes):  # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，这里是 （i：1，lane）
                # skip the class label beyond consideration
                # 跳过不考虑的类标签
                if gt_labels[i] <= 0 or gt_labels[i] > self.num_class:
                    continue
                # 论文中的两个矩阵变化
                P_g2im = projection_g2im(cam_pitch, cam_height, self.K)
                M = np.matmul(self.H_crop, P_g2im)

                # update transformation with image augmentation
                # 使用图像增强更新变换
                if self.data_aug:
                    M = np.matmul(aug_mat, M)
                x_2d, y_2d = projective_transformation(M, lane[:, 0],
                                                       lane[:, 1], lane[:, 2])
                # 通过opencv的坐标来画出车道线
                for j in range(len(x_2d) - 1):
                    label_map = cv2.line(label_map,
                                         (int(x_2d[j]), int(y_2d[j])), (int(x_2d[j + 1]), int(y_2d[j + 1])),
                                         color=np.float64(gt_labels[i]), thickness=3)  # color=np.asscalar(gt_labels[i])

            # new
            # label_map num=187
            # path_1 = '/home/kaai/PycharmProjects/pythonProject2/Pytorch_Generalized_3D_Lane_Detection-master/Apollo_Sim_3D_Lane_Release/segment'
            # label_save = cv2.imwrite(path_1 + str(j) + '.jpg', label_map)
            # print(label_save)
            #
            # label_map = torch.from_numpy(label_map.astype(np.int32)).contiguous().long()

            # if self.transform:
            #     image, label = self.transform((image, label))
            #     image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
            #     label = torch.from_numpy(label).contiguous().long()

            return image, label_map, idx


def homography_crop_resize(org_img_size, crop_y, resize_img_size):
    """
        compute the homography matrix transform original image to cropped and resized image
        计算单应矩阵将原始图像转换为裁剪和调整大小的图像
    :param org_img_size: [org_h, org_w]
    :param crop_y:
    :param resize_img_size: [resize_h, resize_w]
    :return:
    """
    # transform original image region to network input region
    ratio_x = resize_img_size[1] / org_img_size[1]
    ratio_y = resize_img_size[0] / (org_img_size[0] - crop_y)
    H_c = np.array([[ratio_x, 0, 0],
                    [0, ratio_y, -ratio_y*crop_y],
                    [0, 0, 1]])
    return H_c

def projective_transformation(Matrix, x, y, z):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x4 projection matrix
            x (array): original x coordinates
            y (array): original y coordinates
            z (array): original z coordinates
    """
    ones = np.ones((1, len(z)))
    # np.vstack():在竖直方向上堆叠
    # np.vstack [1,2,3] 和 [4,5,6] = [1 2 3 4 5 6]
    coordinates = np.vstack((x, y, z, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/trans[2, :]
    y_vals = trans[1, :]/trans[2, :]
    return x_vals, y_vals

def data_aug_rotate(img):
    # assume img in PIL image format
    rot = random.uniform(-np.pi/18, np.pi/18)
    # 返回a,b之间的随机浮点数，若a<=b则范围[a,b]，若a>=b则范围[b,a] ，a和b可以是实数
    # rot = random.uniform(-10, 10)
    center_x = img.width / 2
    center_y = img.height / 2
    # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), rot, 1.0)
    img_rot = np.array(img)
    #warpAffine ：意思是仿射变化
    #flags – 插值方法的组合（int 类型！）
    img_rot = cv2.warpAffine(img_rot, rot_mat, (img.width, img.height), flags=cv2.INTER_LINEAR)
    # img_rot = img.rotate(rot)
    # rot = rot / 180 * np.pi
    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
    return img_rot,rot_mat

def projective_transformation(Matrix, x, y, z):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x4 projection matrix
            x (array): original x coordinates
            y (array): original y coordinates
            z (array): original z coordinates
    """
    ones = np.ones((1, len(z)))
    # np.vstack():在竖直方向上堆叠
    # np.vstack [1,2,3] 和 [4,5,6] = [1 2 3 4 5 6]
    coordinates = np.vstack((x, y, z, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/trans[2, :]
    y_vals = trans[1, :]/trans[2, :]
    return x_vals, y_vals

def projection_g2im(cam_pitch, cam_height, K):
    #P_g2C as the rotation matrix
    #K indicating camera intrinsic parameters
    P_g2c = np.array([[1,                             0,                              0,          0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                      [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0]])
    P_g2im = np.matmul(K, P_g2c) # P_g2c * K
    return P_g2im