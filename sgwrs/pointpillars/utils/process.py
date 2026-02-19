import copy
import numba
import numpy as np
import random
import torch
import pdb
import pyvista as pv
from pointpillars.ops.iou3d_module import boxes_overlap_bev, boxes_iou_bev
import math


def setup_seed(seed=0, deterministic = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def bbox_camera2lidar(bboxes, tr_velo_to_cam, r0_rect):
    '''
    bboxes: shape=(N, 7)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    return: shape=(N, 7)
    '''
    x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    xyz_size = np.concatenate([z_size, x_size, y_size], axis=1)
    extended_xyz = np.pad(bboxes[:, :3], ((0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = np.linalg.inv(r0_rect @ tr_velo_to_cam)
    xyz = extended_xyz @ rt_mat.T
    bboxes_lidar = np.concatenate([xyz[:, :3], xyz_size, bboxes[:, 6:]], axis=1)
    return np.array(bboxes_lidar, dtype=np.float32)


def bbox_lidar2camera(bboxes, tr_velo_to_cam, r0_rect):
    '''
    bboxes: shape=(N, 7)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    return: shape=(N, 7)
    '''
    x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    xyz_size = np.concatenate([y_size, z_size, x_size], axis=1)
    extended_xyz = np.pad(bboxes[:, :3], ((0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = r0_rect @ tr_velo_to_cam
    xyz = extended_xyz @ rt_mat.T
    bboxes_camera = np.concatenate([xyz[:, :3], xyz_size, bboxes[:, 6:]], axis=1)
    return bboxes_camera


def points_camera2image(points, P2):
    '''
    points: shape=(N, 8, 3) 
    P2: shape=(4, 4)
    return: shape=(N, 8, 2)
    '''
    extended_points = np.pad(points, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1.0) # (n, 8, 4)
    image_points = extended_points @ P2.T # (N, 8, 4)
    image_points = image_points[:, :, :2] / image_points[:, :, 2:3]
    return image_points


def points_lidar2image(points, tr_velo_to_cam, r0_rect, P2):
    '''
    points: shape=(N, 8, 3) 
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    P2: shape=(4, 4)
    return: shape=(N, 8, 2)
    '''
    # points = points[:, :, [1, 2, 0]]
    extended_points = np.pad(points, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1.0) # (N, 8, 4)
    rt_mat = r0_rect @ tr_velo_to_cam
    camera_points = extended_points @ rt_mat.T # (N, 8, 4)
    # camera_points = camera_points[:, :, [1, 2, 0, 3]]
    image_points = camera_points @ P2.T # (N, 8, 4)
    image_points = image_points[:, :, :2] / image_points[:, :, 2:3]

    return image_points


def points_camera2lidar(points, tr_velo_to_cam, r0_rect):
    '''
    points: shape=(N, 8, 3) 
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    return: shape=(N, 8, 3)
    '''
    extended_xyz = np.pad(points, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = np.linalg.inv(r0_rect @ tr_velo_to_cam)
    xyz = extended_xyz @ rt_mat.T
    return xyz[..., :3]


def bbox3d2bevcorners(bboxes):
    '''
    bboxes: shape=(n, 7)

                ^ x (-0.5 * pi)
                |
                |                (bird's eye view)
       (-pi)  o |
        y <-------------- (0)
                 \ / (ag)
                  \ 
                   \ 

    return: shape=(n, 4, 2)
    '''
    centers, dims, angles = bboxes[:, :2], bboxes[:, 3:5], bboxes[:, 6]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bev_corners = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], dtype=np.float32)
    bev_corners = bev_corners[None, ...] * dims[:, None, :] # (1, 4, 2) * (n, 1, 2) -> (n, 4, 2)

    # 2. rotate
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # in fact, -angle
    rot_mat = np.array([[rot_cos, rot_sin], 
                        [-rot_sin, rot_cos]]) # (2, 2, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0)) # (N, 2, 2)
    bev_corners = bev_corners @ rot_mat # (n, 4, 2)

    # 3. translate to centers
    bev_corners += centers[:, None, :] 
    return bev_corners.astype(np.float32)


def bbox3d2corners(bboxes):
    '''
    bboxes: shape=(n, 7)
    return: shape=(n, 8, 3)
           ^ z   x            6 ------ 5
           |   /             / |     / |
           |  /             2 -|---- 1 |   
    y      | /              |  |     | | 
    <------|o               | 7 -----| 4
                            |/   o   |/    
                            3 ------ 0 
    x: front, y: left, z: top
    '''
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bboxes_corners = np.array([[-0.5, -0.5, 0], [-0.5, -0.5, 1.0], [-0.5, 0.5, 1.0], [-0.5, 0.5, 0.0],
                               [0.5, -0.5, 0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [0.5, 0.5, 0.0]], 
                               dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :] # (1, 8, 3) * (n, 1, 3) -> (n, 8, 3)

    # 2. rotate around z axis
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # in fact, -angle
    rot_mat = np.array([[rot_cos, rot_sin, np.zeros_like(rot_cos)],
                        [-rot_sin, rot_cos, np.zeros_like(rot_cos)],
                        [np.zeros_like(rot_cos), np.zeros_like(rot_cos), np.ones_like(rot_cos)]], 
                        dtype=np.float32) # (3, 3, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0)) # (n, 3, 3)
    bboxes_corners = bboxes_corners @ rot_mat # (n, 8, 3)

    # 3. translate to centers
    bboxes_corners += centers[:, None, :]
    return bboxes_corners


def bbox3d2corners_camera(bboxes):
    '''
    bboxes: shape=(n, 7)
    return: shape=(n, 8, 3)
        z (front)            6 ------ 5
        /                  / |     / |
       /                  2 -|---- 1 |   
      /                   |  |     | | 
    |o ------> x(right)   | 7 -----| 4
    |                     |/   o   |/    
    |                     3 ------ 0 
    |
    v y(down)                   
    '''
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bboxes_corners = np.array([[0.5, 0.0, -0.5], [0.5, -1.0, -0.5], [-0.5, -1.0, -0.5], [-0.5, 0.0, -0.5],
                               [0.5, 0.0, 0.5], [0.5, -1.0, 0.5], [-0.5, -1.0, 0.5], [-0.5, 0.0, 0.5]], 
                               dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :] # (1, 8, 3) * (n, 1, 3) -> (n, 8, 3)

    # 2. rotate around y axis
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # in fact, angle
    rot_mat = np.array([[rot_cos, np.zeros_like(rot_cos), rot_sin],
                        [np.zeros_like(rot_cos), np.ones_like(rot_cos), np.zeros_like(rot_cos)],
                        [-rot_sin, np.zeros_like(rot_cos), rot_cos]], 
                        dtype=np.float32) # (3, 3, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0)) # (n, 3, 3)
    bboxes_corners = bboxes_corners @ rot_mat # (n, 8, 3)

    # 3. translate to centers
    bboxes_corners += centers[:, None, :]
    return bboxes_corners


def group_rectangle_vertexs(bboxes_corners):
    '''
    bboxes_corners: shape=(n, 8, 3)
    return: shape=(n, 6, 4, 3)
    '''
    rec1 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 1], bboxes_corners[:, 3], bboxes_corners[:, 2]], axis=1) # (n, 4, 3)
    rec2 = np.stack([bboxes_corners[:, 4], bboxes_corners[:, 7], bboxes_corners[:, 6], bboxes_corners[:, 5]], axis=1) # (n, 4, 3)
    rec3 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 4], bboxes_corners[:, 5], bboxes_corners[:, 1]], axis=1) # (n, 4, 3)
    rec4 = np.stack([bboxes_corners[:, 2], bboxes_corners[:, 6], bboxes_corners[:, 7], bboxes_corners[:, 3]], axis=1) # (n, 4, 3)
    rec5 = np.stack([bboxes_corners[:, 1], bboxes_corners[:, 5], bboxes_corners[:, 6], bboxes_corners[:, 2]], axis=1) # (n, 4, 3)
    rec6 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 3], bboxes_corners[:, 7], bboxes_corners[:, 4]], axis=1) # (n, 4, 3)
    group_rectangle_vertexs = np.stack([rec1, rec2, rec3, rec4, rec5, rec6], axis=1)
    return group_rectangle_vertexs


@numba.jit(nopython=True)
def bevcorner2alignedbbox(bev_corners):
    '''
    bev_corners: shape=(N, 4, 2)
    return: shape=(N, 4)
    '''
    # xmin, xmax = np.min(bev_corners[:, :, 0], axis=-1), np.max(bev_corners[:, :, 0], axis=-1)
    # ymin, ymax = np.min(bev_corners[:, :, 1], axis=-1), np.max(bev_corners[:, :, 1], axis=-1)

    # why we don't implement like the above ? please see
    # https://numba.pydata.org/numba-doc/latest/reference/numpysupported.html#calculation
    n = len(bev_corners)
    alignedbbox = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        cur_bev = bev_corners[i]
        alignedbbox[i, 0] = np.min(cur_bev[:, 0])
        alignedbbox[i, 2] = np.max(cur_bev[:, 0])
        alignedbbox[i, 1] = np.min(cur_bev[:, 1])
        alignedbbox[i, 3] = np.max(cur_bev[:, 1])
    return alignedbbox


# modified from https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/pipelines/data_augment_utils.py#L31
@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    """Box collision test.
    Args:
        boxes (np.ndarray): Corners of current boxes. # (n1, 4, 2)
        qboxes (np.ndarray): Boxes to be avoid colliding. # (n2, 4, 2)
        clockwise (bool, optional): Whether the corners are in
            clockwise order. Default: True.
    return: shape=(n1, n2)
    """
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack((boxes, boxes[:, slices, :]),
                           axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = bevcorner2alignedbbox(boxes)
    qboxes_standup = bevcorner2alignedbbox(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = (
                min(boxes_standup[i, 2], qboxes_standup[j, 2]) -
                max(boxes_standup[i, 0], qboxes_standup[j, 0]))
            if iw > 0:
                ih = (
                    min(boxes_standup[i, 3], qboxes_standup[j, 3]) -
                    max(boxes_standup[i, 1], qboxes_standup[j, 1]))
                if ih > 0:
                    for k in range(4):
                        for box_l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, box_l, 0]
                            D = lines_qboxes[j, box_l, 1]
                            acd = (D[1] - A[1]) * (C[0] -
                                                   A[0]) > (C[1] - A[1]) * (
                                                       D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] -
                                                   B[0]) > (C[1] - B[1]) * (
                                                       D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (
                                        C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (
                                        D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for box_l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (
                                    boxes[i, k, 0] - qboxes[j, box_l, 0])
                                cross -= vec[0] * (
                                    boxes[i, k, 1] - qboxes[j, box_l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for box_l in range(4):  # point box_l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (
                                        qboxes[j, k, 0] - boxes[i, box_l, 0])
                                    cross -= vec[0] * (
                                        qboxes[j, k, 1] - boxes[i, box_l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret


def group_plane_equation(bbox_group_rectangle_vertexs):
    '''
    bbox_group_rectangle_vertexs: shape=(n, 6, 4, 3)
    return: shape=(n, 6, 4)
    '''
    # 1. generate vectors for a x b
    vectors = bbox_group_rectangle_vertexs[:, :, :2] - bbox_group_rectangle_vertexs[:, :, 1:3]
    normal_vectors = np.cross(vectors[:, :, 0], vectors[:, :, 1]) # (n, 6, 3)
    normal_d = np.einsum('ijk,ijk->ij', bbox_group_rectangle_vertexs[:, :, 0], normal_vectors) # (n, 6)
    plane_equation_params = np.concatenate([normal_vectors, -normal_d[:, :, None]], axis=-1)
    return plane_equation_params


@numba.jit(nopython=True)
def points_in_bboxes(points, plane_equation_params):
    '''
    points: shape=(N, 3)
    plane_equation_params: shape=(n, 6, 4)
    return: shape=(N, n), bool
    '''
    N, n = len(points), len(plane_equation_params)
    m = plane_equation_params.shape[1]
    masks = np.ones((N, n), dtype=np.bool_)
    for i in range(N):
        x, y, z = points[i, :3]
        for j in range(n):
            bbox_plane_equation_params = plane_equation_params[j]
            for k in range(m):
                a, b, c, d = bbox_plane_equation_params[k]
                if a * x + b * y + c * z + d >= 0:
                    masks[i][j] = False
                    break
    return masks


def remove_pts_in_bboxes(points, bboxes, rm=True):
    '''
    points: shape=(N, 3)
    bboxes: shape=(n, 7)
    return: shape=(N, n), bool
    '''
    # 1. get 6 groups of rectangle vertexs
    bboxes_corners = bbox3d2corners(bboxes) # (n, 8, 3)
    bbox_group_rectangle_vertexs = group_rectangle_vertexs(bboxes_corners) # (n, 6, 4, 3)

    # 2. calculate plane equation: ax + by + cd + d = 0
    group_plane_equation_params = group_plane_equation(bbox_group_rectangle_vertexs)

    # 3. Judge each point inside or outside the bboxes
    # if point (x0, y0, z0) lies on the direction of normal vector(a, b, c), then ax0 + by0 + cz0 + d > 0.
    masks = points_in_bboxes(points, group_plane_equation_params) # (N, n)

    if not rm:
        return masks
        
    # 4. remove point insider the bboxes
    masks = np.any(masks, axis=-1)

    return points[~masks]


# modified from https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/structures/utils.py#L11
def limit_period(val, offset=0.5, period=np.pi):
    """
    val: array or float
    offset: float
    period: float
    return: Value in the range of [-offset * period, (1-offset) * period]
    """
    limited_val = val - np.floor(val / period + offset) * period
    return limited_val


def nearest_bev(bboxes):
    '''
    bboxes: (n, 7), (x, y, z, w, l, h, theta)
    return: (n, 4), (x1, y1, x2, y2)
    '''    
    bboxes_bev = copy.deepcopy(bboxes[:, [0, 1, 3, 4]])
    bboxes_angle = limit_period(bboxes[:, 6].cpu(), offset=0.5, period=np.pi).to(bboxes_bev)
    bboxes_bev = torch.where(torch.abs(bboxes_angle[:, None]) > np.pi / 4, bboxes_bev[:, [0, 1, 3, 2]], bboxes_bev)
    
    bboxes_xy = bboxes_bev[:, :2]
    bboxes_wl = bboxes_bev[:, 2:]
    bboxes_bev_x1y1x2y2 = torch.cat([bboxes_xy - bboxes_wl / 2, bboxes_xy + bboxes_wl / 2], dim=-1)
    return bboxes_bev_x1y1x2y2


def iou2d(bboxes1, bboxes2, metric=0):
    '''
    bboxes1: (n, 4), (x1, y1, x2, y2)
    bboxes2: (m, 4), (x1, y1, x2, y2)
    return: (n, m)
    '''
    bboxes_x1 = torch.maximum(bboxes1[:, 0][:, None], bboxes2[:, 0][None, :]) # (n, m)
    bboxes_y1 = torch.maximum(bboxes1[:, 1][:, None], bboxes2[:, 1][None, :]) # (n, m)
    bboxes_x2 = torch.minimum(bboxes1[:, 2][:, None], bboxes2[:, 2][None, :])
    bboxes_y2 = torch.minimum(bboxes1[:, 3][:, None], bboxes2[:, 3][None, :])

    bboxes_w = torch.clamp(bboxes_x2 - bboxes_x1, min=0)
    bboxes_h = torch.clamp(bboxes_y2 - bboxes_y1, min=0)

    iou_area = bboxes_w * bboxes_h # (n, m)
    
    bboxes1_wh = bboxes1[:, 2:] - bboxes1[:, :2]
    area1 = bboxes1_wh[:, 0] * bboxes1_wh[:, 1] # (n, )
    bboxes2_wh = bboxes2[:, 2:] - bboxes2[:, :2]
    area2 = bboxes2_wh[:, 0] * bboxes2_wh[:, 1] # (m, )
    if metric == 0:
        iou = iou_area / (area1[:, None] + area2[None, :] - iou_area + 1e-8)
    elif metric == 1:
        iou = iou_area / (area1[:, None] + 1e-8)
    return iou


def iou2d_nearest(bboxes1, bboxes2):
    '''
    bboxes1: (n, 7), (x, y, z, w, l, h, theta)
    bboxes2: (m, 7),
    return: (n, m)
    '''
    bboxes1_bev = nearest_bev(bboxes1)
    bboxes2_bev = nearest_bev(bboxes2)
    iou = iou2d(bboxes1_bev, bboxes2_bev)
    return iou


def iou3d(bboxes1, bboxes2):
    '''
    bboxes1: (n, 7), (x, y, z, w, l, h, theta)
    bboxes2: (m, 7)
    return: (n, m)
    '''
    # 1. height overlap
    bboxes1_bottom, bboxes2_bottom = bboxes1[:, 2], bboxes2[:, 2] # (n, ), (m, )
    bboxes1_top, bboxes2_top = bboxes1[:, 2] + bboxes1[:, 5], bboxes2[:, 2] + bboxes2[:, 5] # (n, ), (m, )
    bboxes_bottom = torch.maximum(bboxes1_bottom[:, None], bboxes2_bottom[None, :]) # (n, m) 
    bboxes_top = torch.minimum(bboxes1_top[:, None], bboxes2_top[None, :])
    height_overlap =  torch.clamp(bboxes_top - bboxes_bottom, min=0)

    # 2. bev overlap
    bboxes1_x1y1 = bboxes1[:, :2] - bboxes1[:, 3:5] / 2
    bboxes1_x2y2 = bboxes1[:, :2] + bboxes1[:, 3:5] / 2
    bboxes2_x1y1 = bboxes2[:, :2] - bboxes2[:, 3:5] / 2
    bboxes2_x2y2 = bboxes2[:, :2] + bboxes2[:, 3:5] / 2
    bboxes1_bev = torch.cat([bboxes1_x1y1, bboxes1_x2y2, bboxes1[:, 6:]], dim=-1)
    bboxes2_bev = torch.cat([bboxes2_x1y1, bboxes2_x2y2, bboxes2[:, 6:]], dim=-1)
    bev_overlap = boxes_overlap_bev(bboxes1_bev, bboxes2_bev) # (n, m)

    # 3. overlap and volume
    overlap = height_overlap * bev_overlap
    volume1 = bboxes1[:, 3] * bboxes1[:, 4] * bboxes1[:, 5]
    volume2 = bboxes2[:, 3] * bboxes2[:, 4] * bboxes2[:, 5]
    volume = volume1[:, None] + volume2[None, :] # (n, m)

    # 4. iou
    iou = overlap / (volume - overlap + 1e-8)

    return iou
    

def iou3d_camera(bboxes1, bboxes2):
    '''
    bboxes1: (n, 7), (x, y, z, w, l, h, theta)
    bboxes2: (m, 7)
    return: (n, m)
    '''
    # 1. height overlap
    bboxes1_bottom, bboxes2_bottom = bboxes1[:, 1] - bboxes1[:, 4], bboxes2[:, 1] -  bboxes2[:, 4] # (n, ), (m, )
    bboxes1_top, bboxes2_top = bboxes1[:, 1], bboxes2[:, 1] # (n, ), (m, )
    bboxes_bottom = torch.maximum(bboxes1_bottom[:, None], bboxes2_bottom[None, :]) # (n, m) 
    bboxes_top = torch.minimum(bboxes1_top[:, None], bboxes2_top[None, :])
    height_overlap =  torch.clamp(bboxes_top - bboxes_bottom, min=0)

    # 2. bev overlap
    bboxes1_x1y1 = bboxes1[:, [0, 2]] - bboxes1[:, [3, 5]] / 2
    bboxes1_x2y2 = bboxes1[:, [0, 2]] + bboxes1[:, [3, 5]] / 2
    bboxes2_x1y1 = bboxes2[:, [0, 2]] - bboxes2[:, [3, 5]] / 2
    bboxes2_x2y2 = bboxes2[:, [0, 2]] + bboxes2[:, [3, 5]] / 2
    bboxes1_bev = torch.cat([bboxes1_x1y1, bboxes1_x2y2, bboxes1[:, 6:]], dim=-1)
    bboxes2_bev = torch.cat([bboxes2_x1y1, bboxes2_x2y2, bboxes2[:, 6:]], dim=-1)
    bev_overlap = boxes_overlap_bev(bboxes1_bev, bboxes2_bev) # (n, m)

    # 3. overlap and volume
    overlap = height_overlap * bev_overlap
    volume1 = bboxes1[:, 3] * bboxes1[:, 4] * bboxes1[:, 5]
    volume2 = bboxes2[:, 3] * bboxes2[:, 4] * bboxes2[:, 5]
    volume = volume1[:, None] + volume2[None, :] # (n, m)

    # 4. iou
    iou = overlap / (volume - overlap + 1e-8)

    return iou


def iou_bev(bboxes1, bboxes2):
    '''
    bboxes1: (n, 5), (x, z, w, h, theta)
    bboxes2: (m, 5)
    return: (n, m)
    '''
    bboxes1_x1y1 = bboxes1[:, :2] - bboxes1[:, 2:4] / 2
    bboxes1_x2y2 = bboxes1[:, :2] + bboxes1[:, 2:4] / 2
    bboxes2_x1y1 = bboxes2[:, :2] - bboxes2[:, 2:4] / 2
    bboxes2_x2y2 = bboxes2[:, :2] + bboxes2[:, 2:4] / 2
    bboxes1_bev = torch.cat([bboxes1_x1y1, bboxes1_x2y2, bboxes1[:, 4:]], dim=-1)
    bboxes2_bev = torch.cat([bboxes2_x1y1, bboxes2_x2y2, bboxes2[:, 4:]], dim=-1)
    bev_overlap = boxes_iou_bev(bboxes1_bev, bboxes2_bev) # (n, m)

    return bev_overlap


def keep_bbox_from_image_range(result, tr_velo_to_cam, r0_rect, P2, image_shape):
    '''
    result: dict(lidar_bboxes, labels, scores)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    P2: shape=(4, 4)
    image_shape: (h, w)
    return: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    '''
    h, w = image_shape

    lidar_bboxes = result['lidar_bboxes']
    labels = result['labels']
    scores = result['scores']
    camera_bboxes = bbox_lidar2camera(lidar_bboxes, tr_velo_to_cam, r0_rect) # (n, 7)
    bboxes_points = bbox3d2corners_camera(camera_bboxes) # (n, 8, 3)
    image_points = points_camera2image(bboxes_points, P2) # (n, 8, 2)
    image_x1y1 = np.min(image_points, axis=1) # (n, 2)
    image_x1y1 = np.maximum(image_x1y1, 0)
    image_x2y2 = np.max(image_points, axis=1) # (n, 2)
    image_x2y2 = np.minimum(image_x2y2, [w, h])
    bboxes2d = np.concatenate([image_x1y1, image_x2y2], axis=-1)

    keep_flag = (image_x1y1[:, 0] < w) & (image_x1y1[:, 1] < h) & (image_x2y2[:, 0] > 0) & (image_x2y2[:, 1] > 0)
    
    result = {
        'lidar_bboxes': lidar_bboxes[keep_flag],
        'labels': labels[keep_flag],
        'scores': scores[keep_flag],
        'bboxes2d': bboxes2d[keep_flag],
        'camera_bboxes': camera_bboxes[keep_flag]
    }
    return result


def keep_bbox_from_lidar_range(result, pcd_limit_range):
    '''
    result: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    pcd_limit_range: []
    return: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    '''
    lidar_bboxes, labels, scores = result['lidar_bboxes'], result['labels'], result['scores']
    if 'bboxes2d' not in result:
        result['bboxes2d'] = np.zeros_like(lidar_bboxes[:, :4])
    if 'camera_bboxes' not in result:
        result['camera_bboxes'] = np.zeros_like(lidar_bboxes)
    bboxes2d, camera_bboxes = result['bboxes2d'], result['camera_bboxes']
    flag1 = lidar_bboxes[:, :3] > pcd_limit_range[:3][None, :] # (n, 3)
    flag2 = lidar_bboxes[:, :3] < pcd_limit_range[3:][None, :] # (n, 3)
    keep_flag = np.all(flag1, axis=-1) & np.all(flag2, axis=-1)
    
    result = {
        'lidar_bboxes': lidar_bboxes[keep_flag],
        'labels': labels[keep_flag],
        'scores': scores[keep_flag],
        'bboxes2d': bboxes2d[keep_flag],
        'camera_bboxes': camera_bboxes[keep_flag]
    }
    return result

# Use image corners instead of shape
def keep_bbox_from_image_range_2(result, tr_velo_to_cam, r0_rect, P2, image_corners):
    '''
    result: dict(lidar_bboxes, labels, scores)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    P2: shape=(4, 4)                                0-------3
    image_corners: (0(x,y),1(x,y),2(x,y),3(x,y))    |       |
                                                    1-------2
    return: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    '''

    s_h = int(min(image_corners[0][1], image_corners[2][1]))
    s_w = int(min(image_corners[0][0], image_corners[2][0]))
    h = int(abs(image_corners[0][1] - image_corners[2][1]))
    w = int(abs(image_corners[0][0] - image_corners[2][0])) 

    lidar_bboxes = result['lidar_bboxes']
    labels = result['labels']
    scores = result['scores']
    camera_bboxes = bbox_lidar2camera(lidar_bboxes, tr_velo_to_cam, r0_rect) # (n, 7)
    bboxes_points = bbox3d2corners_camera(camera_bboxes) # (n, 8, 3)
    image_points = points_camera2image(bboxes_points, P2) # (n, 8, 2)
    image_x1y1_o = np.min(image_points, axis=1) # (n, 2)
    image_x1y1 = np.maximum(image_x1y1_o, [s_w, s_h])
    image_x2y2_o = np.max(image_points, axis=1) # (n, 2)
    image_x2y2 = np.minimum(image_x2y2_o, [s_w + w, s_h + h])
    bboxes2d = np.concatenate([image_x1y1, image_x2y2], axis=-1)

    keep_flag = (image_x1y1[:, 0] < s_w + w) & (image_x1y1[:, 1] < s_h + h) & (image_x2y2[:, 0] > s_w) & (image_x2y2[:, 1] > s_h)
    
    result = {
        'lidar_bboxes': lidar_bboxes[keep_flag],
        'labels': labels[keep_flag],
        'scores': scores[keep_flag],
        'bboxes2d': bboxes2d[keep_flag],
        'camera_bboxes': camera_bboxes[keep_flag]
    }

    return result

def points_in_bboxes_v2(points, r0_rect, tr_velo_to_cam, dimensions, location, rotation_y, name):
    '''
    points: shape=(N, 4) 
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    dimensions: shape=(n, 3) 
    location: shape=(n, 3) 
    rotation_y: shape=(n, ) 
    name: shape=(n, )
    return:
        indices: shape=(N, n_valid_bbox), indices[i, j] denotes whether point i is in bbox j. 
        n_total_bbox: int. 
        n_valid_bbox: int, not including 'DontCare' 
        bboxes_lidar: shape=(n_valid_bbox, 7) 
        name: shape=(n_valid_bbox, )
    '''
    n_total_bbox = len(dimensions)
    n_valid_bbox = len([item for item in name if item != 'DontCare'])
    location, dimensions = location[:n_valid_bbox], dimensions[:n_valid_bbox]
    rotation_y, name = rotation_y[:n_valid_bbox], name[:n_valid_bbox]
    bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=1)
    bboxes_lidar = bbox_camera2lidar(bboxes_camera, tr_velo_to_cam, r0_rect)
    bboxes_corners = bbox3d2corners(bboxes_lidar)
    group_rectangle_vertexs_v = group_rectangle_vertexs(bboxes_corners)
    frustum_surfaces = group_plane_equation(group_rectangle_vertexs_v)
    indices = points_in_bboxes(points[:, :3], frustum_surfaces) # (N, n), N is points num, n is bboxes number
    return indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name


def get_points_num_in_bbox(points, r0_rect, tr_velo_to_cam, dimensions, location, rotation_y, name):
    '''
    points: shape=(N, 4) 
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    dimensions: shape=(n, 3) 
    location: shape=(n, 3) 
    rotation_y: shape=(n, ) 
    name: shape=(n, )
    return: shape=(n, )
    '''
    indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name = \
        points_in_bboxes_v2(
            points=points, 
            r0_rect=r0_rect, 
            tr_velo_to_cam=tr_velo_to_cam, 
            dimensions=dimensions, 
            location=location, 
            rotation_y=rotation_y, 
            name=name)
    points_num = np.sum(indices, axis=0)
    non_valid_points_num = [-1] * (n_total_bbox - n_valid_bbox)
    points_num = np.concatenate([points_num, non_valid_points_num], axis=0)
    return np.array(points_num, dtype=np.int32)


# Modified from https://github.com/open-mmlab/mmdetection3d/blob/f45977008a52baaf97640a0e9b2bbe5ea1c4be34/mmdet3d/core/bbox/box_np_ops.py#L609
def remove_outside_points(points, r0_rect, tr_velo_to_cam, P2, image_shape):
    """Remove points which are outside of image.
    Args:
        points (np.ndarray, shape=[N, 3+dims]): Total points.
        rect (np.ndarray, shape=[4, 4]): Matrix to project points in
            specific camera coordinate (e.g. CAM2) to CAM0.
        Trv2c (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinate to lidar coordinate.
        P2 (p.array, shape=[4, 4]): Intrinsics of Camera2.
        image_shape (list[int]): Shape of image.
    Returns:
        np.ndarray, shape=[N, 3+dims]: Filtered points.
    """
    # 5x faster than remove_outside_points_v1(2ms vs 10ms)
    C, R, T = projection_matrix_to_CRT_kitti(P2)
    image_bbox = [0, 0, image_shape[1], image_shape[0]]
    frustum = get_frustum(image_bbox, C)
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T
    frustum = points_camera2lidar(frustum.T[None, ...], tr_velo_to_cam, r0_rect) # (1, 8, 3)
    group_rectangle_vertexs_v = group_rectangle_vertexs(frustum)
    frustum_surfaces = group_plane_equation(group_rectangle_vertexs_v)
    indices = points_in_bboxes(points[:, :3], frustum_surfaces) # (N, 1)
    points = points[indices.reshape([-1])]
    return points


# Copied from https://github.com/open-mmlab/mmdetection3d/blob/f45977008a52baaf97640a0e9b2bbe5ea1c4be34/mmdet3d/core/bbox/box_np_ops.py#L609
def projection_matrix_to_CRT_kitti(proj):
    """Split projection matrix of kitti.
    P = C @ [R|T]
    C is upper triangular matrix, so we need to inverse CR and use QR
    stable for all kitti camera projection matrix.
    Args:
        proj (p.array, shape=[4, 4]): Intrinsics of camera.
    Returns:
        tuple[np.ndarray]: Splited matrix of C, R and T.
    """

    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T


# Copied from https://github.com/open-mmlab/mmdetection3d/blob/f45977008a52baaf97640a0e9b2bbe5ea1c4be34/mmdet3d/core/bbox/box_np_ops.py#L661
def get_frustum(bbox_image, C, near_clip=0.001, far_clip=100):
    """Get frustum corners in camera coordinates.
    Args:
        bbox_image (list[int]): box in image coordinates.
        C (np.ndarray): Intrinsics.
        near_clip (float, optional): Nearest distance of frustum.
            Defaults to 0.001.
        far_clip (float, optional): Farthest distance of frustum.
            Defaults to 100.
    Returns:
        np.ndarray, shape=[8, 3]: coordinates of frustum corners.
    """
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    z_points = np.array(
        [near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, np.newaxis]
    b = bbox_image
    box_corners = np.array(
        [[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]],
        dtype=C.dtype)
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=C.dtype)
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far_clip, -fkv / far_clip], dtype=C.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners],
                            axis=0)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    return ret_xyz

# Return the image corners as well
def get_frustum_2(bbox_image, C, near_clip=0.001, far_clip=100):
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    z_points = np.array(
        [near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, np.newaxis]
    b = bbox_image
    box_corners = np.array(
        [[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]],
        dtype=C.dtype)
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=C.dtype)
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far_clip, -fkv / far_clip], dtype=C.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners],
                            axis=0)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    return ret_xyz, box_corners

# Remove points which are outside the act segment
def iterative_crop(points, r0_rect, tr_velo_to_cam, P2, image_shape, segments, part, target):
    """Remove points which are outside of image.
    Args:
        points (np.ndarray, shape=[N, 3+dims]): Total points. 
        rect (np.ndarray, shape=[4, 4]): Matrix to project points in
            specific camera coordinate (e.g. CAM2) to CAM0.
        Trv2c (np.ndarray, shape=[4, 4]): Matrix to project points in 
            camera coordinate to lidar coordinate. 
        P2 (p.array, shape=[4, 4]): Intrinsics of Camera2. 
        image_shape (list[int]): Shape of image. 
        segments (int): how many parts the pointcloud should be split into
        part (int): Choose which segment to create (ex. for segments=3 : 0 - left, 1 - middle, 2 - right)
        target (int): The minimum amount of points a segment should have
    Returns:
        np.ndarray, shape=[N, 3+dims]: Filtered points. 
    """

    # Can't reach target if we have less points available
    if target>=points.shape[0]:
        raise ValueError

    C, R, T = projection_matrix_to_CRT_kitti(P2)
    tr_x = (image_shape[1] / segments)

    # Couldn't find a better way to start the 'while' without duplicating all that code below
    # Still ugly but I think it's more readable
    start = True

    # We will initialize from a segment that's smaller than 1/n
    overlap = image_shape[1] / int((1.5*segments))  
    new_points=points
    while new_points.shape[0]<target or start:
        start = False

        # If the part is the first or last segment it expands the frustum from the edge (see: if, else)
        # otherwise we expand in both direction but half as quickly so we don't overshoot(see: elif)
        if part == 0:
            image_bbox = [0, 0 , overlap, image_shape[0]]
        elif part<=(segments-1):
            image_bbox = [0 + tr_x*part + tr_x/2 - (overlap/2), 0 , tr_x*part + tr_x/2 + (overlap/2), image_shape[0]]
        else:
            image_bbox = [image_shape[1] - overlap, 0 , image_shape[1], image_shape[0]]

        # Frustum calculations from original 'remove_outside_points'
        # Forgot the equations but it involves camera intrinsics 
        # Useful article if you want to get what it's doing:
        # https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec 
        frustum, img_corners = get_frustum_2(image_bbox, C)
        frustum -= T
        frustum = np.linalg.inv(R) @ frustum.T
        frustum = points_camera2lidar(frustum.T[None, ...], tr_velo_to_cam, r0_rect) # (1, 8, 3)
        group_rectangle_vertexs_v = group_rectangle_vertexs(frustum)
        frustum_surfaces = group_plane_equation(group_rectangle_vertexs_v)
        indices = points_in_bboxes(points[:, :3], frustum_surfaces) # (N, 1)

        # Update new_points to be the new slice and increase how much to grow based on distance to our target
        new_points = points[indices.reshape([-1])]
        overlap = overlap + (target - new_points.shape[0])/100

    # Fix PC size with RS (some PC size is target+1, target+2, etc ...)
    # It will remove just very few points (1-5 point)
    new_points = new_points[np.random.choice(new_points.shape[0], target, replace=False)]

    return new_points, frustum, img_corners

def calc_truncation (frustum_corners, bbox_corners):
    # Assuming 'frustum_corners' and 'bbox_corners' are (8, 3) numpy arrays
    # Create triangulated mesh with pyvista

    frustum_mesh = pv.PolyData(frustum_corners).delaunay_3d()
    bbox_mesh = pv.PolyData(bbox_corners).delaunay_3d()

    # Calculate how much of the box is outside the frustum
    clipped_bbox = bbox_mesh.clip_surface(frustum_mesh)
    original_volume = bbox_mesh.volume
    clipped_volume = clipped_bbox.volume
    # Calculate truncation. Since truncation is a percentage (0.0 to 1.0)
    # take 100% - (ratio of how much is cut off from it all) = Percentage that remains inside
    truncation = 1 - (clipped_volume / original_volume)
    # rounding for consistency with Kitti ground truth format
    truncation = round(truncation, 2)

    return truncation

def fps(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 4] (x, y, z, intensity).
        npoint: number of samples
    Return:
        centroids: sampled pointcloud data, [B, npoint, 4].
    """
    device = xyz.device
    
    B, N, C = xyz.shape  # B: Batch size, N: Number of points, C: 4 (x, y, z, intensity)
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10  # Initialize large distances
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # Random initial farthest point
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)  # Use all columns (x, y, z, intensity)
        dist = torch.sum((xyz[:, :, :3] - centroid[:, :, :3]) ** 2, -1)  # Compute Euclidean distance (ignore intensity)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]

    # Select the actual points corresponding to the sampled indices
    sampled_points = xyz[batch_indices.unsqueeze(-1), centroids]
    
    return sampled_points

# = Weighted Random Sampling ==========================================================================================

def cart2sph(x, y, z):
    """
    Cartesian -> Spherical coordinates (azimuth, elevation, range)
    All inputs and outputs are torch tensors.
    """
    azimuth = torch.atan2(y, x)
    range_ = torch.sqrt(x**2 + y**2 + z**2)
    elevation = torch.arcsin(z / torch.clamp(range_, min=1e-8))
    return azimuth, elevation, range_

def weighted_random_sample_no_replacement(weights, num_samples):
    """
    Tensor-based weighted random sampling without replacement using Gumbel-max trick.
    """
    weights = weights.float()
    num_elements = weights.shape[0]
    num_samples = min(num_samples, num_elements)

    if num_samples == 0 or num_elements == 0:
        return torch.empty(0, dtype=torch.long, device=weights.device)

    if torch.sum(weights) == 0:
        return torch.randperm(num_elements, device=weights.device)[:num_samples]

    log_weights = torch.log(weights + 1e-12)
    gumbel_noise = -torch.log(-torch.log(torch.rand(num_elements, device=weights.device)))
    scores = log_weights + gumbel_noise
    _, indices = torch.topk(scores, num_samples, largest=True)
    return indices

def hybrid_spherical_downsample(points, max_num_points, *,
                                 origin=None,
                                 min_azimuth=None,
                                 max_azimuth=None,
                                 min_elevation=None,
                                 max_elevation=None,
                                 weighting_exponent=1.0,
                                 ds_ratio=1):
    """
    Downsample a 3D point cloud using hybrid spherical binning and weighted cell sampling.
    
    Args:
        points (torch.Tensor): (N, C) tensor where C >= 3 (x, y, z, [optional features])
        max_num_points (int): Number of output points to keep
    Returns:
        downsampled_points (torch.Tensor): (M, C)
        downsampled_indices (torch.LongTensor): (M,)
    """

    azimuth_resolution = 0.33 * math.log2(ds_ratio)
    elevation_resolution = 0.33 * math.log2(ds_ratio)
    num_range_bins = 600 / math.log2(ds_ratio)

    #print(azimuth_resolution)
    #print(elevation_resolution)
    #print(num_range_bins)
    #exit()

    device = points.device
    dtype = points.dtype
    num_channels = points.shape[1]

    if num_channels < 3:
        raise ValueError("Input 'points' must have at least 3 dimensions (x, y, z).")

    if points.shape[0] == 0 or max_num_points == 0:
        return torch.zeros((0, num_channels), device=device, dtype=dtype), torch.empty(0, dtype=torch.long, device=device)

    if max_num_points >= points.shape[0]:
        return points.clone(), torch.arange(points.shape[0], device=device)

    if origin is None:
        origin = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
    else:
        origin = origin.to(device=device, dtype=dtype)

    xyz = points[:, :3]
    points_xyz = xyz - origin
    azimuth_rad, elevation_rad, ranges = cart2sph(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2])
    azimuth_deg = torch.rad2deg(azimuth_rad)
    elevation_deg = torch.rad2deg(elevation_rad)

    min_azimuth = torch.floor(torch.min(azimuth_deg)) if min_azimuth is None else min_azimuth
    max_azimuth = torch.ceil(torch.max(azimuth_deg)) if max_azimuth is None else max_azimuth
    min_elevation = torch.floor(torch.min(elevation_deg)) if min_elevation is None else min_elevation
    max_elevation = torch.ceil(torch.max(elevation_deg)) if max_elevation is None else max_elevation

    if min_azimuth >= max_azimuth:
        max_azimuth = min_azimuth + azimuth_resolution
    if min_elevation >= max_elevation:
        max_elevation = min_elevation + elevation_resolution

    num_az_bins = max(1, int(torch.ceil((max_azimuth - min_azimuth) / azimuth_resolution)))
    num_el_bins = max(1, int(torch.ceil((max_elevation - min_elevation) / elevation_resolution)))
    max_range_val = torch.max(ranges)
    grid_size_range = max_range_val / num_range_bins if max_range_val > 0 else torch.finfo(dtype).eps

    azimuth_norm = torch.deg2rad(azimuth_deg - min_azimuth)
    elevation_norm = torch.deg2rad(elevation_deg - min_elevation)
    grid_size_azimuth = torch.deg2rad((max_azimuth - min_azimuth) / num_az_bins)
    grid_size_elevation = torch.deg2rad((max_elevation - min_elevation) / num_el_bins)

    az_idx = torch.clamp((azimuth_norm / grid_size_azimuth).long(), 0, num_az_bins - 1)
    el_idx = torch.clamp((elevation_norm / grid_size_elevation).long(), 0, num_el_bins - 1)
    r_idx = torch.clamp((ranges / grid_size_range).long(), 0, num_range_bins - 1)

    linear_cell_idx = az_idx * (num_el_bins * num_range_bins) + el_idx * num_range_bins + r_idx

    # GPU-optimalizált egyedi cellák és hozzárendelés
    unique_cells, inverse_idx, counts = torch.unique(linear_cell_idx, return_inverse=True, return_counts=True)

    eps = torch.finfo(dtype).eps
    weights = 1.0 / (counts.float() ** weighting_exponent + eps)

    num_available = unique_cells.shape[0]
    num_cells_to_select = min(max_num_points, num_available)

    if num_cells_to_select == 0:
        return torch.zeros((0, num_channels), device=device, dtype=dtype), torch.empty(0, dtype=torch.long, device=device)

    selected_indices = weighted_random_sample_no_replacement(weights, num_cells_to_select)
    selected_cells = unique_cells[selected_indices]

    # GPU-s maszkolás és első előfordulás megtalálása
    selection_mask = (linear_cell_idx[:, None] == selected_cells[None, :])  # [N, K]
    first_indices = selection_mask.float().argmax(dim=0)  # [K]
    return points[first_indices], first_indices
