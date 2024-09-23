import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet3d.datasets.pipelines import LoadPointsFromFile
from mmdet3d.core.bbox import box_np_ops as box_np_ops
import torch
import cv2


@PIPELINES.register_module()
class PointToMultiViewDepth(object):
    def __init__(self, downsample=1, min_dist=1e-5, max_dist=None):
        self.downsample = downsample
        self.min_dist = min_dist
        self.max_dist = max_dist

    def points2depthmap(self, points, height, width, img, cid):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        depth_map_mask = torch.zeros((height, width), dtype=torch.bool)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]

        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) \
                & (coor[:, 1] >= 0) & (coor[:, 1] < height) \
                & (depth >= self.min_dist)
        if self.max_dist is not None:
            kept1 = kept1 & (depth < self.max_dist)
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + 1 - depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        depth_map_mask[coor[:, 1], coor[:, 0]] = True

        return depth_map, depth_map_mask

    def __call__(self, results):
        imgs = results['img']
        pts = results['points'].tensor[:, :3]
        lidar2img_rt = results['lidar2img']
        pts = torch.cat(
            [pts, torch.ones((pts.shape[0], 1), dtype=pts.dtype)], -1)
        lidar2img_rt = torch.tensor(lidar2img_rt, dtype=pts.dtype)
        depth_map_list = []
        depth_map_mask_list = []
        for cid in range(len(imgs)):
            points_img = pts.matmul(lidar2img_rt[cid].T)
            points_img[:, :2] /= points_img[:, 2:3]
            depth_map, depth_mask_map = self.points2depthmap(points_img, imgs[cid].shape[0],
                                                             imgs[cid].shape[1], imgs[cid], cid)
            depth_map_list.append(depth_map)
            depth_map_mask_list.append(depth_mask_map)

        depth_map = torch.stack(depth_map_list)
        depth_map_mask = torch.stack(depth_map_mask_list)
        results['depth_map'] = depth_map
        results['depth_map_mask'] = depth_map_mask
        return results

@PIPELINES.register_module()
class CustomLoadPointsFromFile(LoadPointsFromFile):
    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        super(CustomLoadPointsFromFile, self).__init__(coord_type, load_dim, use_dim, shift_height, use_color, file_client_args)

    def _load(self, pts_filename):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)

        return points

    def __call__(self, results):
        pts_filename = results['pts_filename']
        points = self._load(pts_filename)
        results['points'] = points

        # next_pts_filename = results['next_pts_filename']
        # if next_pts_filename is not None:
        #     next_points = self._load(next_pts_filename)
        # else:
        #     next_points = None
        # results['next_points'] = next_points
        return results


@PIPELINES.register_module()
class TemporalPointToMultiViewDepth(object):
    def __init__(self, downsample=1, min_dist=1e-5, max_dist=None):
        self.downsample = downsample
        self.min_dist = min_dist
        self.max_dist = max_dist

    def points2depthmap(self, points_moved_img, origin_points_img, foreground_mask, height, width, img, cid):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        offset_map = torch.zeros((height, width, 3), dtype=torch.float32)
        depth_map_mask = torch.zeros((height, width), dtype=torch.bool)
        foreground_map_mask = torch.zeros((height, width), dtype=torch.bool)

        coor_offset = (origin_points_img[:, :2] - points_moved_img[:, :2]) / self.downsample
        coor = torch.round(points_moved_img[:, :2] / self.downsample)
        next_coor = torch.round(origin_points_img[:, :2] / self.downsample)
        depth = points_moved_img[:, 2]
        next_depth = origin_points_img[:, 2]

        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) \
                & (coor[:, 1] >= 0) & (coor[:, 1] < height) \
                & (depth >= self.min_dist)
        # kept1_next = (next_coor[:, 0] >= 0) & (next_coor[:, 0] < width) \
        #         & (next_coor[:, 1] >= 0) & (next_coor[:, 1] < height) \
        #         & (next_depth >= self.min_dist)
        # kept1 = kept1 & kept1_next
        # if self.max_dist is not None:
        #     kept1 = kept1 & (depth < self.max_dist)

        coor, depth = coor[kept1], depth[kept1]
        next_coor, next_depth = next_coor[kept1], next_depth[kept1]
        coor_offset = coor_offset[kept1]
        foreground_mask = foreground_mask[kept1]

        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + 1 - depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        next_coor, next_depth = next_coor[sort], next_depth[sort]
        coor_offset = coor_offset[sort]
        foreground_mask = foreground_mask[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        next_coor, next_depth = next_coor[kept2], next_depth[kept2]
        coor_offset = coor_offset[kept2]
        foreground_mask = foreground_mask[kept2]

        coor = coor.to(torch.long)
        next_coor = next_coor.to(torch.long)

        depth_map[coor[:, 1], coor[:, 0]] = depth
        # depth_map[next_coor[:, 1], next_coor[:, 0]] = next_depth
        depth_map_mask[coor[:, 1], coor[:, 0]] = True
        offset_map[coor[:, 1], coor[:, 0], 0:2] = coor_offset
        offset_map[coor[:, 1], coor[:, 0], 2] = next_depth - depth
        # offset_map[..., 0] /= width
        # offset_map[..., 1] /= height
        # offset_map[..., 2] /= (self.max_dist - self.min_dist)
        foreground_map_mask[coor[:, 1], coor[:, 0]] = foreground_mask

        # cv2.imwrite(f"/data/jhhou/code/3dppe/debug_save/depth_raw_img_{cid}.png", img)
        if False:
            # cv2.imwrite(f"/data/jhhou/code/3dppe/debug_save/depth_raw_img_{cid}.png", img)
            blue = np.array([255, 0, 0]).reshape(1, 1, 3)
            red = np.array([0, 0, 255]).reshape(1, 1, 3)
            depth_max = depth_map.max().cpu().numpy()

            img_resize = cv2.resize(img, (depth_map.shape[1], depth_map.shape[0]))
            for ind in range(len(coor)):
                depth_single = depth_map[coor[ind, 1], coor[ind, 0]].cpu().numpy()
                color = blue * depth_single / depth_max + red * (1 - depth_single / depth_max)
                cv2.circle(
                    img_resize,
                    center=(int(coor[ind, 0].cpu().numpy()), int(coor[ind, 1].cpu().numpy())),
                    radius=1,
                    color=(int(color[0, 0, 0]), int(color[0, 0, 1]), int(color[0, 0, 2])),
                    thickness=1,
                )
            cv2.imwrite(f"/data/jhhou/code/3dppe/debug_save/depth_img_{cid}.png", img_resize)

        return depth_map, depth_map_mask, offset_map, foreground_map_mask

    @staticmethod
    def trans_boxes(cur_gt, next_ego_pose, ego_pose, time_diff):
        cur_gt_moved = cur_gt.clone()
        cur_gt_moved[:, 0:2] += time_diff * cur_gt_moved[:, 7:9]

        expand_gt = torch.cat([cur_gt_moved[:, :3], cur_gt_moved.new_ones((cur_gt_moved.shape[0], 1))], axis=-1)

        gt_global = (expand_gt @ ego_pose.T)[:, :3]
        expand_gt_global = torch.cat([gt_global[:, :3], cur_gt_moved.new_ones((gt_global.shape[0], 1))], axis=-1)
        next_gt_center = (expand_gt_global @ np.linalg.inv(next_ego_pose.T))[:, :3]

        cur_gt_moved[:, 0:3] = next_gt_center
        cur_gt_moved[:, 6] = cur_gt_moved[..., 6] + np.arctan2(ego_pose[..., 1, 0], ego_pose[..., 0, 0])
        cur_gt_moved[:, 6] = cur_gt_moved[..., 6] - np.arctan2(next_ego_pose[..., 1, 0], next_ego_pose[..., 0, 0])
        return cur_gt_moved

    @staticmethod
    def trans_points(points, next_ego_pose, ego_pose):
        expand_points = torch.cat([points[:, :3], points.new_ones((points.shape[0], 1))], axis=-1)
        points_global = (expand_points @ next_ego_pose.T)[:, :3]
        expand_points_global = torch.cat([points_global[:, :3], points_global.new_ones((points_global.shape[0], 1))], axis=-1)
        points = (expand_points_global @ np.linalg.inv(ego_pose.T))[:, :3]
        return points

    def __call__(self, results):
        imgs = results['img']
        lidar2img_rt = results['lidar2img']

        lidar2img_rt = torch.tensor(lidar2img_rt, dtype=results['points'].tensor.dtype)
        offset_map_list = list()
        offset_map_mask_list = list()
        foreground_map_mask_list = list()

        with_next = results['next_points'] is not None

        if with_next:
            next_points = results['next_points'].tensor

            x_filt = torch.abs(next_points[:, 0]) < 1
            y_filt = torch.abs(next_points[:, 1]) < 2.5
            next_points = next_points[~(x_filt & y_filt)]

            next_ego_pose = results['next_ego_pose']
            ego_pose = results['ego_pose']
            next_timestamp = results['next_timestamp']
            timestamp = results['timestamp']
            time_diff = next_timestamp - timestamp
            cur_gt = torch.cat([results['gt_bboxes_3d'].gravity_center, results['gt_bboxes_3d'].tensor[:, 3:]], dim=-1)
            next_gt = self.trans_boxes(cur_gt, next_ego_pose, ego_pose, time_diff)
            point_indices = box_np_ops.points_in_rbbox(next_points.numpy(), next_gt[..., 0:7].numpy(), origin=(0.5, 0.5, 0.5))
            foreground_points = list()
            origin_foreground_points = list()
            background_points = next_points[point_indices.sum(-1) == 0]
            for i in range(len(next_gt)):
                origin_box_points = next_points[point_indices[:, i]]
                box_points = origin_box_points.clone()
                if len(box_points) == 0:
                    continue
                else:
                    mask = (abs(next_gt[i][..., 7]) > 0.4) | (abs(next_gt[i][..., 8]) > 0.4)
                    if mask.sum() == 0:
                        box_points[:, 0:2] = origin_box_points[:, 0:2]
                    else:
                        box_points[:, 0:2] -= next_gt[i][..., 7:9] * time_diff
                    foreground_points.append(box_points)
                    origin_foreground_points.append(origin_box_points)
            if len(foreground_points) == 0:
                points_moved = next_points.clone()
                origin_points = next_points.clone()
                foreground_mask = torch.zeros_like(points_moved[..., 0], dtype=torch.bool)
            else:
                foreground_points = torch.cat(foreground_points, dim=0)
                origin_foreground_points = torch.cat(origin_foreground_points, dim=0)
                points_moved = torch.cat([foreground_points, background_points], dim=0)
                origin_points = torch.cat([origin_foreground_points, background_points], dim=0)
                foreground_mask = torch.zeros_like(points_moved[..., 0], dtype=torch.bool)
                foreground_mask[0:len(foreground_points)] = True

            # points_moved = background_points
            # origin_points = background_points

            points_moved = self.trans_points(points_moved, next_ego_pose, ego_pose)
            origin_points = self.trans_points(origin_points, next_ego_pose, ego_pose)

            # np.save(f"/data/jhhou/code/3dppe/debug_save/trans_points", points_moved.numpy())
            # np.save(f"/data/jhhou/code/3dppe/debug_save/origin_points", origin_points.numpy())
            # np.save(f"/data/jhhou/code/3dppe/debug_save/cur_gt", cur_gt.numpy())

            points_moved = torch.cat(
                [points_moved[:, 0:3], torch.ones((points_moved.shape[0], 1), dtype=points_moved.dtype)], -1)
            origin_points = torch.cat(
                [origin_points[:, 0:3], torch.ones((origin_points.shape[0], 1), dtype=origin_points.dtype)], -1)

            for cid in range(len(imgs)):
                points_moved_img = points_moved.matmul(lidar2img_rt[cid].T)
                points_moved_img[:, :2] /= points_moved_img[:, 2:3]
                origin_points_img = origin_points.matmul(lidar2img_rt[cid].T)
                origin_points_img[:, :2] /= origin_points_img[:, 2:3]
                depth_map, offset_map_mask, offset_map, foreground_map_mask = self.points2depthmap(points_moved_img, origin_points_img, foreground_mask, imgs[cid].shape[0],
                                                                 imgs[cid].shape[1], imgs[cid], cid)
                offset_map_list.append(offset_map)
                offset_map_mask_list.append(offset_map_mask)
                foreground_map_mask_list.append(foreground_map_mask)

            offset_map = torch.stack(offset_map_list)
            offset_map_mask = torch.stack(offset_map_mask_list)
            foreground_map_mask = torch.stack(foreground_map_mask_list)

        else:
            height = imgs[0].shape[0] // self.downsample
            width = imgs[0].shape[1] // self.downsample
            offset_map = torch.zeros((len(imgs), height, width, 3), dtype=torch.float32)
            offset_map_mask = torch.zeros((len(imgs), height, width), dtype=torch.float32)
            foreground_map_mask = torch.zeros((len(imgs), height, width), dtype=torch.bool)
        results['offset_map'] = offset_map
        results['offset_map_mask'] = offset_map_mask
        results['foreground_map_mask'] = foreground_map_mask
        return results
