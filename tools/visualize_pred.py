import numpy as np
import cv2
import os
import os.path as ops
import math
import ujson as json
import matplotlib
import sys

from tools.utils import *

min_y = 0
max_y = 80

colors = [[1, 0, 0],  # red
          [0, 1, 0],  # green
          [0, 0, 1],  # blue
          [1, 0, 1],  # purple
          [0, 1, 1],  # cyan
          [1, 0.7, 0]]  # orange


class lane_visualizer(object):
    def __init__(self, args):
        self.dataset_dir = args.dataset_dir
        self.K = args.K
        self.no_centerline = args.no_centerline
        """
            this visualizer use higher resolution than network input for better look
        """
        self.resize_h = args.org_h
        self.resize_w = args.org_w
        # self.resize_h = args.resize_h
        # self.resize_w = args.resize_w
        self.ipm_w = 2*args.ipm_w
        self.ipm_h = 2*args.ipm_h
        self.H_crop = homography_crop_resize([args.org_h, args.org_w], args.crop_y, [self.resize_h, self.resize_w])
        # transformation from ipm to ground region
        self.H_ipm2g = cv2.getPerspectiveTransform(np.float32([[0, 0],
                                                              [self.ipm_w-1, 0],
                                                              [0, self.ipm_h-1],
                                                              [self.ipm_w-1, self.ipm_h-1]]),
                                                   np.float32(args.top_view_region))
        self.H_g2ipm = np.linalg.inv(self.H_ipm2g)

        self.x_min = args.top_view_region[0, 0]
        self.x_max = args.top_view_region[1, 0]
        # self.y_samples = np.linspace(args.anchor_y_steps[0], args.anchor_y_steps[-1], num=100, endpoint=False)
        self.y_samples = np.linspace(min_y, max_y, num=100, endpoint=False)

    def visualize_lanes(self, pred_lanes, raw_file, gt_cam_height, gt_cam_pitch, ax1, ax2, ax3):
        P_g2im = projection_g2im(gt_cam_pitch, gt_cam_height, self.K)
        # P_gt = P_g2im
        P_gt = np.matmul(self.H_crop, P_g2im)
        H_g2im = homograpthy_g2im(gt_cam_pitch, gt_cam_height, self.K)
        H_im2ipm = np.linalg.inv(np.matmul(self.H_crop, np.matmul(H_g2im, self.H_ipm2g)))

        img = cv2.imread(ops.join(self.dataset_dir, raw_file))
        img = cv2.warpPerspective(img, self.H_crop, (self.resize_w, self.resize_h))
        img = img.astype(np.float) / 255
        im_ipm = cv2.warpPerspective(img, H_im2ipm, (self.ipm_w, self.ipm_h))
        im_ipm = np.clip(im_ipm, 0, 1)

        cnt_pred = len(pred_lanes)
        pred_visibility_mat = np.zeros((cnt_pred, 100))
        for i in range(cnt_pred):
            # # ATTENTION: ensure y mono increase before interpolation: but it can reduce size
            # pred_lanes[i] = make_lane_y_mono_inc(np.array(pred_lanes[i]))
            # pred_lane = prune_3d_lane_by_range(np.array(pred_lanes[i]), self.x_min, self.x_max)
            min_y = np.min(np.array(pred_lanes[i])[:, 1])
            max_y = np.max(np.array(pred_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(np.array(pred_lanes[i]), self.y_samples, out_vis=True)
            pred_lanes[i] = np.vstack([x_values, z_values]).T
            pred_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min,
                                                       np.logical_and(x_values <= self.x_max,
                                                                      np.logical_and(self.y_samples >= min_y,
                                                                                     self.y_samples <= max_y)))
            pred_visibility_mat[i, :] = np.logical_and(pred_visibility_mat[i, :], visibility_vec)

        # draw lanes in multiple color
        for i in range(cnt_pred):
            x_values = pred_lanes[i][:, 0]
            z_values = pred_lanes[i][:, 1]
            # if 'gflat' in pred_file or 'ext' in pred_file:
            x_ipm_values, y_ipm_values = transform_lane_g2gflat(gt_cam_height, x_values, self.y_samples, z_values)
            # remove those points with z_values > gt_cam_height, this is only for visualization on top-view
            x_ipm_values = x_ipm_values[np.where(z_values < gt_cam_height)]
            y_ipm_values = y_ipm_values[np.where(z_values < gt_cam_height)]
            # else:  # mean to visualize original anchor's preparation
            #     x_ipm_values = x_values
            #     y_ipm_values = self.y_samples
            x_ipm_values, y_ipm_values = homographic_transformation(self.H_g2ipm, x_ipm_values, y_ipm_values)
            x_ipm_values = x_ipm_values.astype(np.int)
            y_ipm_values = y_ipm_values.astype(np.int)
            x_2d, y_2d = projective_transformation(P_gt, x_values, self.y_samples, z_values)
            x_2d = x_2d.astype(np.int)
            y_2d = y_2d.astype(np.int)

            color = colors[np.mod(i, len(colors))]
            # draw on image
            for k in range(1, x_2d.shape[0]):
                # only draw the visible portion
                if pred_visibility_mat[i, k - 1] and pred_visibility_mat[i, k]:
                    img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color[-1::-1], 10)

            # draw on ipm
            for k in range(1, x_ipm_values.shape[0]):
                # only draw the visible portion
                if pred_visibility_mat[i, k - 1] and pred_visibility_mat[i, k]:
                    im_ipm = cv2.line(im_ipm, (x_ipm_values[k - 1], y_ipm_values[k - 1]), (x_ipm_values[k], y_ipm_values[k]), color[-1::-1], 3)

            # draw in 3d
            ax3.plot(x_values[np.where(pred_visibility_mat[i, :])],
                     self.y_samples[np.where(pred_visibility_mat[i, :])],
                     z_values[np.where(pred_visibility_mat[i, :])], color=color, linewidth=5)
        ax1.imshow(img[:, :, [2, 1, 0]])
        ax2.imshow(im_ipm[:, :, [2, 1, 0]])