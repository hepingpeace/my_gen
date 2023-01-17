import argparse
import errno
import os
import sys

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.init as init
import torch.optim
from torch.optim import lr_scheduler
import os.path as ops
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
plt.rcParams['figure.figsize'] = (35, 30)


def define_args():
    parser = argparse.ArgumentParser(description='Lane_detection_all_objectives')
    # Paths settings
    parser.add_argument('--dataset_name', type=str, help='the dataset name to be used in saving model names')
    parser.add_argument('--data_dir', type=str, help='The path saving train.json and val.json files')
    parser.add_argument('--dataset_dir', type=str, help='The path saving actual data')
    parser.add_argument('--save_path', type=str, default='data_splits/', help='directory to save output')
    # Dataset settings
    parser.add_argument('--org_h', type=int, default=1080, help='height of the original image')
    parser.add_argument('--org_w', type=int, default=1920, help='width of the original image')
    parser.add_argument('--crop_y', type=int, default=0, help='crop from image')
    parser.add_argument('--cam_height', type=float, default=1.55, help='height of camera in meters')
    parser.add_argument('--pitch', type=float, default=3, help='pitch angle of camera to ground in centi degree')
    parser.add_argument('--fix_cam', type=str2bool, nargs='?', const=True, default=False, help='if to use fix camera')
    parser.add_argument('--no_3d', action='store_true', help='if a dataset include laneline 3D attributes')
    parser.add_argument('--no_centerline', action='store_true', help='if a dataset include centerline')
    # 3DLaneNet settings
    parser.add_argument('--mod', type=str, default='3DLaneNet', help='model to train')
    parser.add_argument("--pretrained", type=str2bool, nargs='?', const=True, default=True, help="use pretrained vgg model")
    parser.add_argument("--batch_norm", type=str2bool, nargs='?', const=True, default=True, help="apply batch norm")
    parser.add_argument("--pred_cam", type=str2bool, nargs='?', const=True, default=False, help="use network to predict camera online?")
    parser.add_argument('--ipm_h', type=int, default=208, help='height of inverse projective map (IPM)')
    parser.add_argument('--ipm_w', type=int, default=128, help='width of inverse projective map (IPM)')
    parser.add_argument('--resize_h', type=int, default=360, help='height of the original image')
    parser.add_argument('--resize_w', type=int, default=480, help='width of the original image')
    parser.add_argument('--y_ref', type=float, default=20.0, help='the reference Y distance in meters from where lane association is determined')
    parser.add_argument('--prob_th', type=float, default=0.5, help='probability threshold for selecting output lanes')
    # General model settings
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--nepochs', type=int, default=30, help='total numbers of epochs')
    parser.add_argument('--learning_rate', type=float, default=5*1e-4, help='learning rate')
    parser.add_argument('--no_cuda', action='store_true', help='if gpu available')
    parser.add_argument('--nworkers', type=int, default=0, help='num of threads')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout in network')
    parser.add_argument('--pretrain_epochs', type=int, default=20, help='Number of epochs to perform segmentation pretraining')
    parser.add_argument('--channels_in', type=int, default=3, help='num channels of input image')
    parser.add_argument('--flip_on', action='store_true', help='Random flip input images on?')
    parser.add_argument('--test_mode', action='store_true', help='prevents loading latest saved model')
    parser.add_argument('--start_epoch', type=int, default=0, help='prevents loading latest saved model')
    parser.add_argument('--evaluate', action='store_true', help='only perform evaluation')
    parser.add_argument('--resume', type=str, default='', help='resume latest saved run')
    parser.add_argument('--vgg_mean', type=float, default=[0.485, 0.456, 0.406], help='Mean of rgb used in pretrained model on ImageNet')
    parser.add_argument('--vgg_std', type=float, default=[0.229, 0.224, 0.225], help='Std of rgb used in pretrained model on ImageNet')
    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='adam', help='adam or sgd')
    parser.add_argument('--weight_init', type=str, default='normal', help='normal, xavier, kaiming, orhtogonal weights initialisation')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 weight decay/regularisation on?')
    parser.add_argument('--lr_decay', action='store_true', help='decay learning rate with rule')
    parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=400, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr_policy', default=None, help='learning rate policy: lambda|step|plateau')
    parser.add_argument('--lr_decay_iters', type=int, default=30, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--clip_grad_norm', type=int, default=0, help='performs gradient clipping')

    # CUDNN usage
    # parser.add_argument("--cudnn", type=str2bool, nargs='?', const=True, default=True, help="cudnn optimization active")
    # Tensorboard settings
    # parser.add_argument("--no_tb", type=str2bool, nargs='?', const=True, default=False, help="Use tensorboard logging by tensorflow")

    # Print settings
    parser.add_argument('--print_freq', type=int, default=500, help='padding')
    parser.add_argument('--save_freq', type=int, default=500, help='padding')
    # Skip batch
    parser.add_argument('--list', type=int, nargs='+', default=[954, 2789], help='Images you want to skip')

    return parser

def sim3d_config(args):

    # set dataset parameters
    args.org_h = 1080
    args.org_w = 1920
    args.crop_y = 0
    args.no_centerline = False
    args.no_3d = False
    args.fix_cam = False
    args.pred_cam = False

    # set camera parameters for the test datasets
    args.K = np.array([[2015., 0., 960.],
                       [0., 2015., 540.],
                       [0., 0., 1.]])

    # specify model settings
    """
    paper presented params:
        args.top_view_region = np.array([[-10, 85], [10, 85], [-10, 5], [10, 5]])
        args.anchor_y_steps = np.array([5, 20, 40, 60, 80, 100])
    """
    # args.top_view_region = np.array([[-10, 83], [10, 83], [-10, 3], [10, 3]])
    # args.anchor_y_steps = np.array([3, 5, 10, 20, 40, 60, 80, 100])
    args.top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
    args.anchor_y_steps = np.array([5, 10, 15, 20, 30, 40, 50, 60, 80, 100])
    args.num_y_steps = len(args.anchor_y_steps)

    # initialize with pre-trained vgg weights
    args.pretrained = False
    # apply batch norm in network
    args.batch_norm = True

def homographic_transformation(Matrix, x, y):
    """
    Helper function to transform coordinates defined by transformation matrix
    变换矩阵定义坐标的辅助函数

    Args:
            Matrix (multi dim - array): 3x3 homography matrix
            x (array): original x coordinates
            y (array): original y coordinates
    """
    ones = np.ones((1, len(y)))
    coordinates = np.vstack((x, y, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/trans[2, :]
    y_vals = trans[1, :]/trans[2, :]
    return x_vals, y_vals

def projective_transformation(Matrix, x, y, z):
    """
    Helper function to transform coordinates defined by transformation matrix
    变换矩阵定义坐标的辅助函数
    Args:
            Matrix (multi dim - array): 3x4 projection matrix
            x (array): original x coordinates
            y (array): original y coordinates
            z (array): original z coordinates
    """
    ones = np.ones((1, len(z)))
    coordinates = np.vstack((x, y, z, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/trans[2, :]
    y_vals = trans[1, :]/trans[2, :]
    return x_vals, y_vals


def transform_lane_gflat2g(h_cam, X_gflat, Y_gflat, Z_g):
    """
        Given X coordinates in flat ground space, Y coordinates in flat ground space, and Z coordinates in real 3D ground space
        with projection matrix from 3D ground to flat ground, compute real 3D coordinates X, Y in 3D ground space.
        给出平面空间X坐标、平面空间Y坐标、三维投影矩阵实三维空间Z坐标，计算三维空间实三维坐标X、Y坐标

    :param P_g2gflat: a 3 X 4 matrix transforms lane form 3d ground x,y,z to flat ground x, y
    :param X_gflat: X coordinates in flat ground space
    :param Y_gflat: Y coordinates in flat ground space
    :param Z_g: Z coordinates in real 3D ground space
    :return:
    """

    X_g = X_gflat - X_gflat * Z_g / h_cam
    Y_g = Y_gflat - Y_gflat * Z_g / h_cam

    return X_g, Y_g


def transform_lane_g2gflat(h_cam, X_g, Y_g, Z_g):
    """
        Given X coordinates in flat ground space, Y coordinates in flat ground space, and Z coordinates in real 3D ground space
        with projection matrix from 3D ground to flat ground, compute real 3D coordinates X, Y in 3D ground space.
        给定平坦地面空间中的X坐标、平坦地面空间中的Y坐标、实三维地面空间中的Z坐标利用投影矩阵从三维地面到平面地面，计算三维地面空间中真实的三维坐标X、Y。

    :param P_g2gflat: a 3 X 4 matrix transforms lane form 3d ground x,y,z to flat ground x, y
    :param X_gflat: X coordinates in flat ground space
    :param Y_gflat: Y coordinates in flat ground space
    :param Z_g: Z coordinates in real 3D ground space
    :return:
    """

    X_gflat = X_g * h_cam / (h_cam - Z_g)
    Y_gflat = Y_g * h_cam / (h_cam - Z_g)

    return X_gflat, Y_gflat


class Visualizer:
    def __init__(self, args, vis_folder='val_vis'):
        self.save_path = args.save_path
        self.vis_folder = vis_folder
        self.no_3d = args.no_3d
        self.no_centerline = args.no_centerline
        self.vgg_mean = args.vgg_mean
        self.vgg_std = args.vgg_std
        self.ipm_w = args.ipm_w
        self.ipm_h = args.ipm_h
        self.num_y_steps = args.num_y_steps

        if args.no_3d:
            self.anchor_dim = args.num_y_steps + 1
        else:
            if 'ext' in args.mod:
                self.anchor_dim = 3 * args.num_y_steps + 1
            else:
                self.anchor_dim = 2 * args.num_y_steps + 1

        x_min = args.top_view_region[0, 0]
        x_max = args.top_view_region[1, 0]
        self.anchor_x_steps = np.linspace(x_min, x_max, np.int(args.ipm_w / 8), endpoint=True)
        self.anchor_y_steps = args.anchor_y_steps

        # transformation from ipm to ground region
        H_ipm2g = cv2.getPerspectiveTransform(np.float32([[0, 0],
                                                          [self.ipm_w-1, 0],
                                                          [0, self.ipm_h-1],
                                                          [self.ipm_w-1, self.ipm_h-1]]),
                                              np.float32(args.top_view_region))
        self.H_g2ipm = np.linalg.inv(H_ipm2g)

        # probability threshold for choosing visualize lanes
        self.prob_th = args.prob_th

    def draw_on_img(self, img, lane_anchor, P_g2im, draw_type='laneline', color=[0, 0, 1]):
        """
        :param img: image in numpy array, each pixel in [0, 1] range
        :param lane_anchor: lane anchor in N X C numpy ndarray, dimension in agree with dataloader
        :param P_g2im: projection from ground 3D coordinates to image 2D coordinates
        :param draw_type: 'laneline' or 'centerline' deciding which to draw
        :param color: [r, g, b] color for line,  each range in [0, 1]
        :return:

        :param img:numpy 数组中的图像，每个像素位于 [0,1]
        :param lane_anchor:N X C numpnd数组中的车道锚， numpy ndarray, dimension 与数据加载器一致
        :param P_g2im:从地面三维坐标到图像二维坐标的投影
        ：param  draw_type: 'lane line' 或 'center line' 决定要绘制哪个
        :param color: [r,g,b]color for 行，每个范围在 [0,1] 中
        :::返回:
        """

        for j in range(lane_anchor.shape[0]):
            # draw laneline

            """
            从lane_anchor中取出x_offsets，并加上self.anchor_x_steps[j]，得到x_3d。
            然后根据P_g2im的维度分别进行投影变换，将三维坐标转换为二维坐标。
            最后，使用cv2.line函数在图像上绘制直线，将车道线的起点和终点的二维坐标作为参数
            """
            if draw_type is 'laneline' and lane_anchor[j, self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_3d = x_offsets + self.anchor_x_steps[j]
                if P_g2im.shape[1] is 3:
                    x_2d, y_2d = homographic_transformation(P_g2im, x_3d, self.anchor_y_steps)
                else:
                    z_3d = lane_anchor[j, self.num_y_steps:self.anchor_dim - 1]
                    x_2d, y_2d = projective_transformation(P_g2im, x_3d, self.anchor_y_steps, z_3d)
                x_2d = x_2d.astype(np.int)
                y_2d = y_2d.astype(np.int)
                for k in range(1, x_2d.shape[0]):
                    img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color, 2)

            # draw centerline
            # 上面是绘制车道线，这里是绘制车道中心线，其他的部分基本相同。
            if draw_type is 'centerline' and lane_anchor[j, 2 * self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, self.anchor_dim:self.anchor_dim + self.num_y_steps]
                x_3d = x_offsets + self.anchor_x_steps[j]
                if P_g2im.shape[1] is 3:
                    x_2d, y_2d = homographic_transformation(P_g2im, x_3d, self.anchor_y_steps)
                else:
                    z_3d = lane_anchor[j, self.anchor_dim + self.num_y_steps:2 * self.anchor_dim - 1]
                    x_2d, y_2d = projective_transformation(P_g2im, x_3d, self.anchor_y_steps, z_3d)
                x_2d = x_2d.astype(np.int)
                y_2d = y_2d.astype(np.int)
                for k in range(1, x_2d.shape[0]):
                    img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color, 2)

            # draw the additional centerline for the merging case
            # 这里是绘制扩展的车道中心线
            if draw_type is 'centerline' and lane_anchor[j, 3 * self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, 2 * self.anchor_dim:2 * self.anchor_dim + self.num_y_steps]
                x_3d = x_offsets + self.anchor_x_steps[j]
                if P_g2im.shape[1] is 3:
                    x_2d, y_2d = homographic_transformation(P_g2im, x_3d, self.anchor_y_steps)
                else:
                    z_3d = lane_anchor[j, 2 * self.anchor_dim + self.num_y_steps:3 * self.anchor_dim - 1]
                    x_2d, y_2d = projective_transformation(P_g2im, x_3d, self.anchor_y_steps, z_3d)
                x_2d = x_2d.astype(np.int)
                y_2d = y_2d.astype(np.int)
                for k in range(1, x_2d.shape[0]):
                    img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color, 2)
        print(cv2.imwrite(img))

        return img

    def draw_on_img_new(self, img, lane_anchor, P_g2im, draw_type='laneline', color=[0, 0, 1]):
        """
        :param img: image in numpy array, each pixel in [0, 1] range
        :param lane_anchor: lane anchor in N X C numpy ndarray, dimension in agree with dataloader
        :param P_g2im: projection from ground 3D coordinates to image 2D coordinates 从地面三维坐标投影到图像二维坐标
        :param draw_type: 'laneline' or 'centerline' deciding which to draw
        :param color: [r, g, b] color for line,  each range in [0, 1]
        :return:

        """
        for j in range(lane_anchor.shape[0]):
            # draw laneline
            if draw_type is 'laneline' and lane_anchor[j, self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_3d = x_offsets + self.anchor_x_steps[j]
                if P_g2im.shape[1] is 3:
                    x_2d, y_2d = homographic_transformation(P_g2im, x_3d, self.anchor_y_steps)
                    visibility = np.ones_like(x_2d)
                else:
                    z_3d = lane_anchor[j, self.num_y_steps:2*self.num_y_steps]
                    x_2d, y_2d = projective_transformation(P_g2im, x_3d, self.anchor_y_steps, z_3d)
                    visibility = lane_anchor[j, 2 * self.num_y_steps:3 * self.num_y_steps]
                x_2d = x_2d.astype(np.int)
                y_2d = y_2d.astype(np.int)
                for k in range(1, x_2d.shape[0]):
                    if visibility[k] > self.prob_th:
                        img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color, 2)
                    else:
                        img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), [0, 0, 0], 2)

            # draw centerline
            if draw_type is 'centerline' and lane_anchor[j, 2 * self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, self.anchor_dim:self.anchor_dim + self.num_y_steps]
                x_3d = x_offsets + self.anchor_x_steps[j]
                if P_g2im.shape[1] is 3:
                    x_2d, y_2d = homographic_transformation(P_g2im, x_3d, self.anchor_y_steps)
                    visibility = np.ones_like(x_2d)
                else:
                    z_3d = lane_anchor[j, self.anchor_dim + self.num_y_steps:self.anchor_dim + 2*self.num_y_steps]
                    x_2d, y_2d = projective_transformation(P_g2im, x_3d, self.anchor_y_steps, z_3d)
                    visibility = lane_anchor[j, self.anchor_dim + 2*self.num_y_steps:self.anchor_dim + 3*self.num_y_steps]
                x_2d = x_2d.astype(np.int)
                y_2d = y_2d.astype(np.int)
                for k in range(1, x_2d.shape[0]):
                    if visibility[k] > self.prob_th:
                        img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color, 2)
                    else:
                        img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), [0, 0, 0], 2)

                # draw the additional centerline for the merging case
                if draw_type is 'centerline' and lane_anchor[j, 3 * self.anchor_dim - 1] > self.prob_th:
                    x_offsets = lane_anchor[j, 2 * self.anchor_dim:2 * self.anchor_dim + self.num_y_steps]
                    x_3d = x_offsets + self.anchor_x_steps[j]
                    if P_g2im.shape[1] is 3:
                        x_2d, y_2d = homographic_transformation(P_g2im, x_3d, self.anchor_y_steps)
                        visibility = np.ones_like(x_2d)
                    else:
                        z_3d = lane_anchor[j,
                               2 * self.anchor_dim + self.num_y_steps:2 * self.anchor_dim + 2 * self.num_y_steps]
                        x_2d, y_2d = projective_transformation(P_g2im, x_3d, self.anchor_y_steps, z_3d)
                        visibility = lane_anchor[j,
                                     2 * self.anchor_dim + 2 * self.num_y_steps:2 * self.anchor_dim + 3 * self.num_y_steps]
                    x_2d = x_2d.astype(np.int)
                    y_2d = y_2d.astype(np.int)
                    for k in range(1, x_2d.shape[0]):
                        if visibility[k] > self.prob_th:
                            img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color, 2)
                        else:
                            img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), [0, 0, 0], 2)
            return img

    def draw_on_ipm(self, im_ipm, lane_anchor, draw_type='laneline', color=[0, 0, 1]):
        for j in range(lane_anchor.shape[0]):
            # draw laneline
            if draw_type is 'laneline' and lane_anchor[j, self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, :self.num_y_steps]
                # 将 lane_anchor 矩阵中第 j 行的前 self.num_y_steps 列的值赋给 x_offsets 变量
                x_g = x_offsets + self.anchor_x_steps[j]

                # compute lanelines in ipm view
                # 然后使用 homographic_transformation 函数将 x_g 和 self.anchor_y_steps 转换为图像坐标系中的坐标，得到 x_ipm 和 y_ipm
                x_ipm, y_ipm = homographic_transformation(self.H_g2ipm, x_g, self.anchor_y_steps)
                x_ipm = x_ipm.astype(np.int)
                y_ipm = y_ipm.astype(np.int)
                for k in range(1, x_g.shape[0]):
                    im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]),
                                      (x_ipm[k], y_ipm[k]), color, 1)

            # draw centerline
            if draw_type is 'centerline' and lane_anchor[j, 2 * self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, self.anchor_dim:self.anchor_dim + self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]

                # compute lanelines in ipm view
                x_ipm, y_ipm = homographic_transformation(self.H_g2ipm, x_g, self.anchor_y_steps)
                x_ipm = x_ipm.astype(np.int)
                y_ipm = y_ipm.astype(np.int)
                for k in range(1, x_g.shape[0]):
                    im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]),
                                      (x_ipm[k], y_ipm[k]), color, 1)


            # draw the additional centerline for the merging case
            if draw_type is 'centerline' and lane_anchor[j, 3 * self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, 2 * self.anchor_dim:2 * self.anchor_dim + self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]

                # compute lanelines in ipm view
                x_ipm, y_ipm = homographic_transformation(self.H_g2ipm, x_g, self.anchor_y_steps)
                x_ipm = x_ipm.astype(np.int)
                y_ipm = y_ipm.astype(np.int)
                for k in range(1, x_g.shape[0]):
                    im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]),
                                      (x_ipm[k], y_ipm[k]), color, 1)
        return im_ipm

    def draw_on_ipm_new(self, im_ipm, lane_anchor, draw_type='laneline', color=[0, 0, 1], width=1):
        for j in range(lane_anchor.shape[0]):
            # draw laneline
            if draw_type is 'laneline' and lane_anchor[j, self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    visibility = np.ones_like(x_g)
                else:
                    # 将 lane_anchor 矩阵中第 j 行的第 2 * self.num_y_steps 到第 3 * self.num_y_steps 列赋值给 visibility 变量。
                    visibility = lane_anchor[j, 2*self.num_y_steps:3*self.num_y_steps]
            # draw centerline
            if draw_type is 'centerline' and lane_anchor[j, 2 * self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, self.anchor_dim:self.anchor_dim + self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    visibility = np.ones_like(x_g)
                else:
                    visibility = lane_anchor[j, self.anchor_dim + 2 * self.num_y_steps:self.anchor_dim + 3 * self.num_y_steps]

                # compute lanelines in ipm view
                x_ipm, y_ipm = homographic_transformation(self.H_g2ipm, x_g, self.anchor_y_steps)
                x_ipm = x_ipm.astype(np.int)
                y_ipm = y_ipm.astype(np.int)
                for k in range(1, x_g.shape[0]):
                    if visibility[k] > self.prob_th:
                        im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]),
                                          (x_ipm[k], y_ipm[k]), color, width)
                    else:
                        im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]),
                                          (x_ipm[k], y_ipm[k]), [0, 0, 0], width)

            # draw the additional centerline for the merging case
            if draw_type is 'centerline' and lane_anchor[j, 3*self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, 2*self.anchor_dim:2*self.anchor_dim + self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    visibility = np.ones_like(x_g)
                else:
                    visibility = lane_anchor[j, 2*self.anchor_dim + 2*self.num_y_steps:2*self.anchor_dim + 3*self.num_y_steps]

                # compute lanelines in ipm view
                x_ipm, y_ipm = homographic_transformation(self.H_g2ipm, x_g, self.anchor_y_steps)
                x_ipm = x_ipm.astype(np.int)
                y_ipm = y_ipm.astype(np.int)
                for k in range(1, x_g.shape[0]):
                    if visibility[k] > self.prob_th:
                        im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]),
                                          (x_ipm[k], y_ipm[k]), color, width)
                    else:
                        im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]),
                                          (x_ipm[k], y_ipm[k]), [0, 0, 0], width)
        return im_ipm

    def draw_3d_curves(self, ax, lane_anchor, draw_type='laneline', color=[0, 0, 1]):
        for j in range(lane_anchor.shape[0]):
            # draw laneline
            if draw_type is 'laneline' and lane_anchor[j, self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    z_g = np.zeros_like(x_g)
                else:
                # 使用 ax.plot 函数在3D坐标系中绘制车道线，x_g，self.anchor_y_steps,
                # z_g 为绘制车道线的三维坐标， color 为绘制车道线的颜色
                    z_g = lane_anchor[j, self.num_y_steps:2*self.num_y_steps]
                ax.plot(x_g, self.anchor_y_steps, z_g, color=color)

            # draw centerline
            if draw_type is 'centerline' and lane_anchor[j, 2*self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, self.anchor_dim:self.anchor_dim + self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    z_g = np.zeros_like(x_g)
                else:
                    z_g = lane_anchor[j, self.anchor_dim + self.num_y_steps:self.anchor_dim + 2*self.num_y_steps]
                ax.plot(x_g, self.anchor_y_steps, z_g, color=color)

            # draw the additional centerline for the merging case 为合并案例绘制额外的中心线
            if draw_type is 'centerline' and lane_anchor[j, 3*self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, 2*self.anchor_dim:2*self.anchor_dim + self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    z_g = np.zeros_like(x_g)
                else:
                    z_g = lane_anchor[j, 2*self.anchor_dim + self.num_y_steps:2*self.anchor_dim + 2*self.num_y_steps]
                ax.plot(x_g, self.anchor_y_steps, z_g, color=color)

    def draw_3d_curves_new(self, ax, lane_anchor, h_cam, draw_type='laneline', color=[0, 0, 1]):
        for j in range(lane_anchor.shape[0]):
            # draw laneline
            if draw_type is 'laneline' and lane_anchor[j, self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_gflat = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    z_g = np.zeros_like(x_gflat)
                    visibility = np.ones_like(x_gflat)
                else:
                    z_g = lane_anchor[j, self.num_y_steps:2*self.num_y_steps]
                    visibility = lane_anchor[j, 2*self.num_y_steps:3*self.num_y_steps]
                x_gflat = x_gflat[np.where(visibility > self.prob_th)]
                z_g = z_g[np.where(visibility > self.prob_th)]

                if len(x_gflat) > 0:
                    # transform lane detected in flat ground space to 3d ground space
                    # 在平坦地面空间检测到转换车道到三维地面空间
                    x_g, y_g = transform_lane_gflat2g(h_cam,
                                                      x_gflat,
                                                      self.anchor_y_steps[np.where(visibility > self.prob_th)],
                                                      z_g)
                    ax.plot(x_g, y_g, z_g, color=color)

            # draw centerline
            if draw_type is 'centerline' and lane_anchor[j, 2*self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, self.anchor_dim:self.anchor_dim + self.num_y_steps]
                x_gflat = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    z_g = np.zeros_like(x_gflat)
                    visibility = np.ones_like(x_gflat)
                else:
                    z_g = lane_anchor[j, self.anchor_dim + self.num_y_steps:self.anchor_dim + 2*self.num_y_steps]
                    visibility = lane_anchor[j, self.anchor_dim + 2*self.num_y_steps:self.anchor_dim + 3*self.num_y_steps]
                x_gflat = x_gflat[np.where(visibility > self.prob_th)]
                z_g = z_g[np.where(visibility > self.prob_th)]
                if len(x_gflat) > 0:
                    # transform lane detected in flat ground space to 3d ground space
                    x_g, y_g = transform_lane_gflat2g(h_cam,
                                                      x_gflat,
                                                      self.anchor_y_steps[np.where(visibility > self.prob_th)],
                                                      z_g)
                    ax.plot(x_g, y_g, z_g, color=color)

            # draw the additional centerline for the merging case
            if draw_type is 'centerline' and lane_anchor[j, 3*self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, 2*self.anchor_dim:2*self.anchor_dim + self.num_y_steps]
                x_gflat = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    z_g = np.zeros_like(x_gflat)
                    visibility = np.ones_like(x_gflat)
                else:
                    z_g = lane_anchor[j, 2*self.anchor_dim + self.num_y_steps:2*self.anchor_dim + 2*self.num_y_steps]
                    visibility = lane_anchor[j, 2*self.anchor_dim + 2*self.num_y_steps:2*self.anchor_dim + 3*self.num_y_steps]
                x_gflat = x_gflat[np.where(visibility > self.prob_th)]
                z_g = z_g[np.where(visibility > self.prob_th)]
                if len(x_gflat) > 0:
                    # transform lane detected in flat ground space to 3d ground space
                    x_g, y_g = transform_lane_gflat2g(h_cam,
                                                      x_gflat,
                                                      self.anchor_y_steps[np.where(visibility > self.prob_th)],
                                                      z_g)
                    ax.plot(x_g, y_g, z_g, color=color)

    def save_result(self, dataset, train_or_val, epoch, batch_i, idx, images, gt, pred, pred_cam_pitch, pred_cam_height, aug_mat=np.identity(3, dtype=np.float), evaluate=False):
        if not dataset.data_aug:
            # aug_mat 是一个二维数组， idx 是一个一维数组。
            # np.expand_dims(aug_mat, axis=0) 函数是将 aug_mat 的维度扩展到一个新的维度，即在第0维处加入一个新维度。
           aug_mat = np.repeat(np.expand_dims(aug_mat, axis=0), idx.shape[0], axis=0)

        for i in range(idx.shape[0]):
            # during training, only visualize the first sample of this batch
            # 在训练期间,请只看本批次的第一批样本。
            if i > 0 and not evaluate:
                break
            im = images.permute(0, 2, 3, 1).data.cpu().numpy()[i]
            # the vgg_std and vgg_mean are for images in [0, 1] range
            # 是针对 [0,1] 范围内的图像的
            im = im * np.array(self.vgg_std)
            im = im + np.array(self.vgg_mean)
            im = np.clip(im, 0, 1)

            gt_anchors = gt[i]
            pred_anchors = pred[i]

            # apply nms to avoid output directly neighbored lanes  应用nms以避免输出直接相邻的车道
            # consider w/o centerline cases 考虑w/o中心线案例
            if self.no_centerline:
                pred_anchors[:, -1] = nms_1d(pred_anchors[:, -1])
            else:
                pred_anchors[:, self.anchor_dim - 1] = nms_1d(pred_anchors[:, self.anchor_dim - 1])
                pred_anchors[:, 2 * self.anchor_dim - 1] = nms_1d(pred_anchors[:, 2 * self.anchor_dim - 1])
                pred_anchors[:, 3 * self.anchor_dim - 1] = nms_1d(pred_anchors[:, 3 * self.anchor_dim - 1])

            H_g2im, P_g2im, H_crop, H_im2ipm = dataset.transform_mats(idx[i])
            if self.no_3d:
                P_gt = np.matmul(H_crop, H_g2im)
                H_g2im_pred = homograpthy_g2im(pred_cam_pitch[i],
                                               pred_cam_height[i], dataset.K)
                P_pred = np.matmul(H_crop, H_g2im_pred)

                # consider data augmentation 考虑到数据增强
                P_gt = np.matmul(aug_mat[i, :, :], P_gt)
                P_pred = np.matmul(aug_mat[i, :, :], P_pred)
            else:
                P_gt = np.matmul(H_crop, P_g2im)
                P_g2im_pred = projection_g2im(pred_cam_pitch[i],
                                              pred_cam_height[i], dataset.K)
                P_pred = np.matmul(H_crop, P_g2im_pred)

                # consider data augmentation 考虑到数据增强
                P_gt = np.matmul(aug_mat[i, :, :], P_gt)
                P_pred = np.matmul(aug_mat[i, :, :], P_pred)

            # update transformation with image augmentation
            H_im2ipm = np.matmul(H_im2ipm, np.linalg.inv(aug_mat[i, :, :]))
            im_ipm = cv2.warpPerspective(im, H_im2ipm, (self.ipm_w, self.ipm_h))
            # 这里可以展现图片
            im_ipm = np.clip(im_ipm, 0, 1)

            # draw lanes on image 画出车道的图片在图片上
            im_laneline = im.copy()
            im_laneline = self.draw_on_img(im_laneline, gt_anchors, P_gt, 'laneline', [0, 0, 1])
            im_laneline = self.draw_on_img(im_laneline, pred_anchors, P_pred, 'laneline', [1, 0, 0])
            if not self.no_centerline:
                im_centerline = im.copy()
                im_centerline = self.draw_on_img(im_centerline, gt_anchors, P_gt, 'centerline', [0, 0, 1])
                im_centerline = self.draw_on_img(im_centerline, pred_anchors, P_pred, 'centerline', [1, 0, 0])

            # draw lanes on ipm 画出出到在ipm上
            ipm_laneline = im_ipm.copy()
            ipm_laneline = self.draw_on_ipm(ipm_laneline, gt_anchors, 'laneline', [0, 0, 1])
            ipm_laneline = self.draw_on_ipm(ipm_laneline, pred_anchors, 'laneline', [1, 0, 0])
            # 没有中心车道线的
            if not self.no_centerline:
                ipm_centerline = im_ipm.copy()
                ipm_centerline = self.draw_on_ipm(ipm_centerline, gt_anchors, 'centerline', [0, 0, 1])
                ipm_centerline = self.draw_on_ipm(ipm_centerline, pred_anchors, 'centerline', [1, 0, 0])

            # plot on a single figure 画出一个图表出来
            if self.no_centerline and self.no_3d:
                fig = plt.figure()
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                ax1.imshow(im_laneline)
                ax2.imshow(ipm_laneline)
            elif not self.no_centerline and self.no_3d:
                fig = plt.figure()
                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222)
                ax3 = fig.add_subplot(223)
                ax4 = fig.add_subplot(224)
                ax1.imshow(im_laneline)
                ax2.imshow(ipm_laneline)
                ax3.imshow(im_centerline)
                ax4.imshow(ipm_centerline)
            elif not self.no_centerline and not self.no_3d:
                fig = plt.figure()
                ax1 = fig.add_subplot(231)
                ax2 = fig.add_subplot(232)
                ax3 = fig.add_subplot(233, projection='3d')
                ax4 = fig.add_subplot(234)
                ax5 = fig.add_subplot(235)
                ax6 = fig.add_subplot(236, projection='3d')
                ax1.imshow(im_laneline)
                ax2.imshow(ipm_laneline)
                self.draw_3d_curves(ax3, gt_anchors, 'laneline', [0, 0, 1])
                self.draw_3d_curves(ax3, pred_anchors, 'laneline', [1, 0, 0])
                ax3.set_xlabel('x axis')
                ax3.set_ylabel('y axis')
                ax3.set_zlabel('z axis')
                bottom, top = ax3.get_zlim()
                ax3.set_zlim(min(bottom, -1), max(top, 1))
                ax3.set_xlim(-20, 20)
                ax3.set_ylim(0, 100)
                ax4.imshow(im_centerline)
                ax5.imshow(ipm_centerline)
                self.draw_3d_curves(ax6, gt_anchors, 'centerline', [0, 0, 1])
                self.draw_3d_curves(ax6, pred_anchors, 'centerline', [1, 0, 0])
                ax6.set_xlabel('x axis')
                ax6.set_ylabel('y axis')
                ax6.set_zlabel('z axis')
                bottom, top = ax6.get_zlim()
                ax6.set_zlim(min(bottom, -1), max(top, 1))
                ax6.set_xlim(-20, 20)
                ax6.set_ylim(0, 100)

            # TODO：更换模式不要保存看一下加遍历图片怎么样
            if evaluate:
                fig.savefig(self.save_path + '/example/' + self.vis_folder + '/infer_{}'.format(idx[i]))
            else:
                fig.savefig(self.save_path + '/example/{}/epoch-{}_batch-{}_idx-{}'.format(train_or_val,
                                                                                           epoch, batch_i, idx[i]))
            plt.clf()
            plt.close(fig)

    def save_result_new(self, dataset, train_or_val, epoch, batch_i, idx, images, gt, pred, pred_cam_pitch, pred_cam_height, aug_mat=np.identity(3, dtype=np.float), evaluate=False):
        if not dataset.data_aug:
            aug_mat = np.repeat(np.expand_dims(aug_mat, axis=0), idx.shape[0], axis=0)

        for i in range(idx.shape[0]):
            # during training, only visualize the first sample of this batch
            if i > 0 and not evaluate:
                break
            im = images.permute(0, 2, 3, 1).data.cpu().numpy()[i]
            # the vgg_std and vgg_mean are for images in [0, 1] range
            im = im * np.array(self.vgg_std)
            im = im + np.array(self.vgg_mean)
            im = np.clip(im, 0, 1)

            gt_anchors = gt[i]
            pred_anchors = pred[i]

            # apply nms to avoid output directly neighbored lanes
            # consider w/o centerline cases
            if self.no_centerline:
                pred_anchors[:, -1] = nms_1d(pred_anchors[:, -1])
            else:
                pred_anchors[:, self.anchor_dim - 1] = nms_1d(pred_anchors[:, self.anchor_dim - 1])
                pred_anchors[:, 2 * self.anchor_dim - 1] = nms_1d(pred_anchors[:, 2 * self.anchor_dim - 1])
                pred_anchors[:, 3 * self.anchor_dim - 1] = nms_1d(pred_anchors[:, 3 * self.anchor_dim - 1])

            H_g2im, P_g2im, H_crop, H_im2ipm = dataset.transform_mats(idx[i])
            P_gt = np.matmul(H_crop, H_g2im)
            H_g2im_pred = homograpthy_g2im(pred_cam_pitch[i],
                                           pred_cam_height[i], dataset.K)
            P_pred = np.matmul(H_crop, H_g2im_pred)

            # consider data augmentation
            P_gt = np.matmul(aug_mat[i, :, :], P_gt)
            P_pred = np.matmul(aug_mat[i, :, :], P_pred)

            # update transformation with image augmentation
            H_im2ipm = np.matmul(H_im2ipm, np.linalg.inv(aug_mat[i, :, :]))
            im_ipm = cv2.warpPerspective(im, H_im2ipm, (self.ipm_w, self.ipm_h))
            im_ipm = np.clip(im_ipm, 0, 1)

            # draw lanes on image
            im_laneline = im.copy()
            im_laneline = self.draw_on_img_new(im_laneline, gt_anchors, P_gt, 'laneline', [0, 0, 1])
            im_laneline = self.draw_on_img_new(im_laneline, pred_anchors, P_pred, 'laneline', [1, 0, 0])
            if not self.no_centerline:
                im_centerline = im.copy()
                im_centerline = self.draw_on_img_new(im_centerline, gt_anchors, P_gt, 'centerline', [0, 0, 1])
                im_centerline = self.draw_on_img_new(im_centerline, pred_anchors, P_pred, 'centerline', [1, 0, 0])

            # draw lanes on ipm
            ipm_laneline = im_ipm.copy()
            ipm_laneline = self.draw_on_ipm_new(ipm_laneline, gt_anchors, 'laneline', [0, 0, 1])
            ipm_laneline = self.draw_on_ipm_new(ipm_laneline, pred_anchors, 'laneline', [1, 0, 0])
            if not self.no_centerline:
                ipm_centerline = im_ipm.copy()
                ipm_centerline = self.draw_on_ipm_new(ipm_centerline, gt_anchors, 'centerline', [0, 0, 1])
                ipm_centerline = self.draw_on_ipm_new(ipm_centerline, pred_anchors, 'centerline', [1, 0, 0])

            # plot on a single figure
            if self.no_centerline and self.no_3d:
                fig = plt.figure()
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                ax1.imshow(im_laneline)
                ax2.imshow(ipm_laneline)
            elif not self.no_centerline and self.no_3d:
                fig = plt.figure()
                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222)
                ax3 = fig.add_subplot(223)
                ax4 = fig.add_subplot(224)
                ax1.imshow(im_laneline)
                ax2.imshow(ipm_laneline)
                ax3.imshow(im_centerline)
                ax4.imshow(ipm_centerline)
            elif not self.no_centerline and not self.no_3d:
                fig = plt.figure()
                ax1 = fig.add_subplot(231)
                ax2 = fig.add_subplot(232)
                ax3 = fig.add_subplot(233, projection='3d')
                ax4 = fig.add_subplot(234)
                ax5 = fig.add_subplot(235)
                ax6 = fig.add_subplot(236, projection='3d')
                ax1.imshow(im_laneline)
                ax2.imshow(ipm_laneline)
                # TODO:use separate gt_cam_height when ready
                self.draw_3d_curves_new(ax3, gt_anchors, pred_cam_height[i], 'laneline', [0, 0, 1])
                self.draw_3d_curves_new(ax3, pred_anchors, pred_cam_height[i], 'laneline', [1, 0, 0])
                ax3.set_xlabel('x axis')
                ax3.set_ylabel('y axis')
                ax3.set_zlabel('z axis')
                bottom, top = ax3.get_zlim()
                ax3.set_xlim(-20, 20)
                ax3.set_ylim(0, 100)
                ax3.set_zlim(min(bottom, -1), max(top, 1))
                ax4.imshow(im_centerline)
                ax5.imshow(ipm_centerline)
                # TODO:use separate gt_cam_height when ready
                self.draw_3d_curves_new(ax6, gt_anchors, pred_cam_height[i], 'centerline', [0, 0, 1])
                self.draw_3d_curves_new(ax6, pred_anchors, pred_cam_height[i], 'centerline', [1, 0, 0])
                ax6.set_xlabel('x axis')
                ax6.set_ylabel('y axis')
                ax6.set_zlabel('z axis')
                bottom, top = ax6.get_zlim()
                ax6.set_xlim(-20, 20)
                ax6.set_ylim(0, 100)
                ax6.set_zlim(min(bottom, -1), max(top, 1))

            if evaluate:
                fig.savefig(self.save_path + '/example/' + self.vis_folder + '/infer_{}'.format(idx[i]))
            else:
                fig.savefig(self.save_path + '/example/{}/epoch-{}_batch-{}_idx-{}'.format(train_or_val,
                                                                                           epoch, batch_i, idx[i]))
            plt.clf()
            plt.close(fig)

def prune_3d_lane_by_visibility(lane_3d, visibility):
    lane_3d = lane_3d[visibility > 0, ...]
    return lane_3d

def prune_3d_lane_by_range(lane_3d, x_min, x_max):
    # TODO: solve hard coded range later
    # remove points with y out of range
    # 3D label may miss super long straight-line with only two points: Not have to be 200, gt need a min-step
    # 2D dataset requires this to rule out those points projected to ground, but out of meaningful range
    # 删除超出范围的点
    # 3D标签可能漏掉超长直线，只有两点: 不必是200，需要迈出一步
    # 2D数据集要求排除那些被投影到地面上的点，但是超出了有意义的范围
    # lane_3d是一个三维数组，np.logical_and(lane_3d[:, 1] > 0, lane_3d[:, 1] < 200)是一个布尔型数组，
    # 表示对第二维的数值进行筛选，只保留大于0且小于200的数值，最后的结果是只保留符合条件的三维数组中的元素。
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 1] > 0, lane_3d[:, 1] < 200), ...]

    # remove lane points out of x range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 0] > x_min,
                                     lane_3d[:, 0] < x_max), ...]
    return lane_3d

def resample_laneline_in_y(input_lane, y_steps, out_vis=False):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
        在每个锚点网格中插入 x,z 值，包括超出输入范围的值
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
                       input_lane:N x 2 或 N x 3nd 数组，一个点为一行（x，y，z-可选）。它要求输入车道的 y 值按升序排列
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range 是否输出只取决于输入范围的能见度指示器
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    y_min = np.min(input_lane[:, 1])-5
    y_max = np.max(input_lane[:, 1])+5

    if input_lane.shape[1] < 3:
        # np.concatenate 是numpy中对array进行拼接的函数
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    # fill_value参数设置为"extrapolate"表示允许插值函数对数组范围以外的值进行插值。
    # np.concatenate 是numpy中对array进行拼接的函数
    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")

    x_values = f_x(y_steps)
    z_values = f_z(y_steps)

    # 首先创建一个output_visibility数组，其元素为y_steps数组中的每一项是否大于等于y_min且小于等于y_max。
    # 然后，使用interp1d函数返回的插值函数f_x和f_z，在y_steps数组的每一项上求出对应的x值和z值，并将其分别存入x_values和z_values数组中，
    # 最后返回x_values，z_values，output_visibility数组。
    # 并且将 output_visibility 转换成float32类型，并加上1e-9，为了防止返回0.
    if out_vis:
        output_visibility = np.logical_and(y_steps >= y_min, y_steps <= y_max)
        return x_values, z_values, output_visibility.astype(np.float32) + 1e-9
    return x_values, z_values

def resample_laneline_in_y_with_vis(input_lane, y_steps, vis_vec):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")
    f_vis = interp1d(input_lane[:, 1], vis_vec, fill_value="extrapolate")

    x_values = f_x(y_steps)
    z_values = f_z(y_steps)
    vis_values = f_vis(y_steps)

    x_values = x_values[vis_values > 0.5]
    y_values = y_steps[vis_values > 0.5]
    z_values = z_values[vis_values > 0.5]
    return np.array([x_values, y_values, z_values]).T

def homography_im2ipm_norm(top_view_region, org_img_size, crop_y, resize_img_size, cam_pitch, cam_height, K):
    """
        Compute the normalized transformation such that image region are mapped to top_view region maps to
        the top view image's 4 corners
        计算归一化转换，使图像区域映射到 top_view 区域映射到
        俯视图图像的四个角
        Ground coordinates: x-right, y-forward, z-up
        地面坐标: x-right, y-forward, z-up
        The purpose of applying normalized transformation应用归一化变换的目的:
        1. invariance in scale change 尺度变化不变性
        2.Torch grid sample is based on normalized grids Tocrch 网格样本是来自归一化网格

    :param top_view_region: a 4 X 2 list of (X, Y) indicating the top-view region corners in order: 4X2 list (X,Y)显示top-view region，顺序如下:
                            top-left, top-right, bottom-left, bottom-right 左上、右上、左下、右下
    :param org_img_size: the size of original image size: [h, w] org_img_size:原始图像大小:[h,w]
    :param crop_y: pixels croped from original img 从原始图像剪切出的像素
    :param resize_img_size: the size of image as network input: [h, w] 图像作为网络输入的大小:[h,w]
    :param cam_pitch: camera pitch angle wrt ground plane 照相机俯仰角写入地平面
    :param cam_height: camera height wrt ground plane in meters 照相机高度写入地面平面（米单位)
    :param K: camera intrinsic parameters 照相机固有参数
    :return: H_im2ipm_norm: the normalized transformation from image to IPM image 从图像到 IPM 图像的归一化转换
    """

    # compute homography transformation from ground to image (only this depends on cam_pitch and cam height)
    H_g2im = homograpthy_g2im(cam_pitch, cam_height, K)
    # transform original image region to network input region
    H_c = homography_crop_resize(org_img_size, crop_y, resize_img_size)
    H_g2im = np.matmul(H_c, H_g2im)

    # compute top-view corners' coordinates in image 计算图像中的顶视图角坐标
    x_2d, y_2d = homographic_transformation(H_g2im, top_view_region[:, 0], top_view_region[:, 1])
    border_im = np.concatenate([x_2d.reshape(-1, 1), y_2d.reshape(-1, 1)], axis=1) # 将两个向量的值画到一起
    # 这里需要可视化一下

    # compute the normalized transformation 计算归一化变换
    border_im[:, 0] = border_im[:, 0] / resize_img_size[1]
    border_im[:, 1] = border_im[:, 1] / resize_img_size[0]
    border_im = np.float32(border_im)
    dst = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
    # img to ipm
    # 它用来将图像从一个坐标系变换到另一个坐标系。cv2.
    # border_im.shape(4,2),dst(2,4)
    # 得到3*3
    H_im2ipm_norm = cv2.getPerspectiveTransform(border_im, dst)
    # ipm to img
    H_ipm2im_norm = cv2.getPerspectiveTransform(dst, border_im)
    return H_im2ipm_norm, H_ipm2im_norm

def homography_ipmnorm2g(top_view_region):
    src = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
    # 透视变换(Perspective Transformation)是将成像投影到一个新的视平面(Viewing Plane)，也称作投影映射(Projective Mapping)。
    H_ipmnorm2g = cv2.getPerspectiveTransform(src, np.float32(top_view_region))
    return H_ipmnorm2g

def homograpthy_g2im(cam_pitch, cam_height, K):
    # transform top-view region to original image region 将顶视图区域转换为原始图像区域
    R_g2c = np.array([[1, 0, 0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch)],
                      [0, np.sin(np.pi / 2 + cam_pitch), np.cos(np.pi / 2 + cam_pitch)]])
    H_g2im = np.matmul(K, np.concatenate([R_g2c[:, 0:2], [[0], [cam_height], [0]]], 1))
    return H_g2im

def projection_g2im(cam_pitch, cam_height, K):
    P_g2c = np.array([[1,                             0,                              0,          0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                      [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0]])
    P_g2im = np.matmul(K, P_g2c) # Matrix product of two arrays.
    return P_g2im

def nms_1d(v):
    """

    :param v: a 1D numpy array
    :return:
    """
    v_out = v.copy()
    len = v.shape[0]
    if len < 2:
        return v
    for i in range(len):
        if i is not 0 and v[i - 1] > v[i]:
            v_out[i] = 0.
        elif i is not len-1 and v[i+1] > v[i]:
            v_out[i] = 0.
    return v_out