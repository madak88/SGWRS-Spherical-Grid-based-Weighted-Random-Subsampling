
import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import sys

CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR)

from pointpillars.utils import \
    read_points, read_calib, read_label, write_points, write_pickle, \
    get_points_num_in_bbox, group_rectangle_vertexs, group_plane_equation, bbox_camera2lidar, bbox3d2corners, points_in_bboxes, \
    iterative_crop, calc_truncation

def judge_difficulty(annotation_dict):
    truncated = annotation_dict['truncated']
    occluded = annotation_dict['occluded']
    bbox = annotation_dict['bbox']
    height = bbox[:, 3] - bbox[:, 1]

    MIN_HEIGHTS = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.30, 0.50]
    difficultys = []
    for h, o, t in zip(height, occluded, truncated):
        difficulty = -1
        for i in range(2, -1, -1):
            if h > MIN_HEIGHTS[i] and o <= MAX_OCCLUSION[i] and t <= MAX_TRUNCATION[i]:
                difficulty = i
        difficultys.append(difficulty)
    return np.array(difficultys, dtype=int)

def create_data_info_pkl(data_root, data_type, prefix, segments=3, target=8192):

    print(f"Processing {data_type} data..")

    sep = os.path.sep
    split = 'training'
    
    ids_file = os.path.join(CUR, 'pointpillars', 'dataset', 'ImageSets', f'{data_type}.txt')
    with open(ids_file, 'r') as f:
        ids = [id.strip() for id in f.readlines()]

    saved_cropped_path = os.path.join(data_root, split, 'velodyne_cropped')
    os.makedirs(saved_cropped_path, exist_ok=True)

    kitti_infos_dict = {}

    for id in tqdm(ids):
    
        img_path = os.path.join(data_root, split, 'image_2', f'{id}.png')
        lidar_path = os.path.join(data_root, split, 'velodyne_reduced', f'{id}.bin')
        calib_path = os.path.join(data_root, split, 'calib', f'{id}.txt')

        img = cv2.imread(img_path)
        image_shape = img.shape[:2]
        lidar_points = read_points(lidar_path)
        calib_dict = read_calib(calib_path)

        point_range=[0, -39.68, -3, 69.12, 39.68, 1]
        flag_x_low = lidar_points[:, 0] > point_range[0]
        flag_y_low = lidar_points[:, 1] > point_range[1]
        flag_z_low = lidar_points[:, 2] > point_range[2]
        flag_x_high = lidar_points[:, 0] < point_range[3]
        flag_y_high = lidar_points[:, 1] < point_range[4]
        flag_z_high = lidar_points[:, 2] < point_range[5]
        keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
        lidar_points = lidar_points[keep_mask]
    
        for part in range(segments):

            cur_info_dict={}
            
            cropped_lidar_points, frustum_points, img_corners = iterative_crop(
                points=lidar_points, 
                r0_rect=calib_dict['R0_rect'], 
                tr_velo_to_cam=calib_dict['Tr_velo_to_cam'], 
                P2=calib_dict['P2'],
                image_shape=image_shape,
                segments = segments,
                part=part,
                target=target
            )

            group_rectangle_vertexs_v = group_rectangle_vertexs(frustum_points)
            frustum = group_plane_equation(group_rectangle_vertexs_v)

            # Labels

            label_path = os.path.join(data_root, split, 'label_2', f'{id}.txt')
            annotation_dict = read_label(label_path)

            dimensions = annotation_dict['dimensions']
            location = annotation_dict['location']
            rotation_y = annotation_dict['rotation_y']
            bboxes = np.concatenate([location, dimensions, rotation_y[:, None]], axis=-1)
            bboxes = bbox_camera2lidar(bboxes, calib_dict['Tr_velo_to_cam'], calib_dict['R0_rect'])
            bbox_corners = bbox3d2corners(bboxes)
            bbox_planes = group_rectangle_vertexs(bbox_corners)
            bbox_planes = group_plane_equation(bbox_planes)

            indices = np.empty(0,dtype=int)
            truncation = np.empty(0,dtype=int)
            for i in range(bbox_corners.shape[0]):
                bbox_mask = bbox_corners[i,:,:]
                corners_inside = points_in_bboxes(bbox_mask, frustum)
                points_in_bbox = points_in_bboxes(cropped_lidar_points, bbox_planes)
                remaining_points = bbox_corners[i,corners_inside.reshape([-1])]
                if len(remaining_points) > 0 and np.count_nonzero(points_in_bbox[:,i]) > 5:
                    truncation = np.append(truncation, calc_truncation(frustum_points[0,:,:],bbox_mask))
                    indices = np.append(indices, i)

            if len(indices) == 0: continue

            total = int(id)*segments+part
            new_id = f'{total:06d}'

            lidar_path = os.path.join(saved_cropped_path, f'{new_id}.bin')
            write_points(cropped_lidar_points, lidar_path)

            cur_info_dict['image'] = {
                'image_shape': image_shape,
                'image_path': sep.join(img_path.split(sep)[-3:]), 
                'image_idx': int(id),
                'image_corners': img_corners # New Image Plane
            }
            cur_info_dict['velodyne_path'] = sep.join(lidar_path.split(sep)[-3:])
            cur_info_dict['calib'] = calib_dict
            
            cur_info_dict['crop'] = {'segments': segments, 'target': target, 'part': part}

            with open(label_path, "r") as f: lines = f.readlines()
            modified_lines = []
            j = 0
            for i in indices:
                pos = lines[i].find('.')
                modified_line = lines[i][:pos-1] + f'{truncation[j]:.2f}' + lines[i][pos+3:]
                modified_lines.append(modified_line)
                j += 1

            annotation_dict = read_label(modified_lines, fromFile=False)
            annotation_dict['difficulty'] = judge_difficulty(annotation_dict)
            annotation_dict['num_points_in_gt'] = get_points_num_in_bbox(
                points=cropped_lidar_points,
                r0_rect=calib_dict['R0_rect'], 
                tr_velo_to_cam=calib_dict['Tr_velo_to_cam'],
                dimensions=annotation_dict['dimensions'],
                location=annotation_dict['location'],
                rotation_y=annotation_dict['rotation_y'],
                name=annotation_dict['name']
            )
            cur_info_dict['annos'] = annotation_dict
            kitti_infos_dict[int(new_id)] = cur_info_dict

    saved_path = os.path.join(data_root, f'{prefix}_infos_{data_type}.pkl')
    write_pickle(kitti_infos_dict, saved_path)
    return kitti_infos_dict

def main(args):
    data_root = args.data_root
    prefix = args.prefix
    ## 1. train: create data infomation pkl file && create reduced point clouds
    kitti_train_infos_dict = create_data_info_pkl(data_root, 'train', prefix)
    ## 2. val: create data infomation pkl file && create reduced point clouds
    kitti_val_infos_dict = create_data_info_pkl(data_root, 'val', prefix)
    ## 3. trainval: create data infomation pkl file
    kitti_trainval_infos_dict = {**kitti_train_infos_dict, **kitti_val_infos_dict}
    saved_path = os.path.join(data_root, f'{prefix}_infos_trainval.pkl')
    write_pickle(kitti_trainval_infos_dict, saved_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset infomation')
    parser.add_argument('--data_root', default='kitti_data', help='your data root for kitti')
    parser.add_argument('--prefix', default='kitti', help='the prefix name for the saved .pkl file')
    args = parser.parse_args()
    main(args)
    