
import open3d as o3d
import numpy as np
import argparse
import os

class Visualizer:
    def __init__(self, pcd_folder, bbox_folder, calib_folder, frames, start_idx, no_bbox):
        self.pcd_folder = pcd_folder
        self.bbox_folder = bbox_folder
        self.calib_folder = calib_folder
        self.frames = frames
        self.current_idx = frames.index(start_idx)
        self.no_bbox = no_bbox
        
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        
        self.load_frame(self.frames[self.current_idx])

    def load_frame(self, frame):
    
        pcd_path = os.path.join(self.pcd_folder, f"{frame}.bin")
        if not self.no_bbox: bbox_path = os.path.join(self.bbox_folder, f"{frame}.txt")
        calib_path = os.path.join(self.calib_folder, f"{frame}.txt")
        
        tr_velo_to_cam, r0_rect = self.load_calib(calib_path)
        points = self.load_point_cloud(pcd_path)
        if not self.no_bbox: bboxes = self.load_bboxes(bbox_path, tr_velo_to_cam, r0_rect)
        
        self.vis.clear_geometries()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        self.vis.add_geometry(pcd)
        
        if not self.no_bbox: 
            for bbox in bboxes: self.vis.add_geometry(bbox)

    def load_calib(self, calib_path):
        with open(calib_path, 'r') as f:
            lines = f.readlines()
        calib = {}
        for line in lines:
            if not line.strip(): continue
            key, value = line.split(':', 1)
            calib[key.strip()] = np.array([float(x) for x in value.strip().split()])
        
        r0_rect = np.eye(4)
        r0_rect[:3, :3] = calib['R0_rect'].reshape(3, 3)
        
        tr_velo_to_cam = np.eye(4)
        tr_velo_to_cam[:3, :4] = calib['Tr_velo_to_cam'].reshape(3, 4)
        
        return tr_velo_to_cam, r0_rect

    def load_point_cloud(self, bin_path):
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        return points

    def load_bboxes(self, txt_path, tr_velo_to_cam, r0_rect):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        bboxes = []
        raw_bboxes = []
        for line in lines:
            if not line.strip() or line.startswith("DontCare"): continue
            data = line.split()
            l, h, w, x, y, z, yaw = map(float, data[-8:-1])
            raw_bboxes.append([x, y, z, h, w, l, yaw])
        raw_bboxes = np.array(raw_bboxes, dtype=np.float32)
        bboxes_lidar = self.bbox_camera2lidar(raw_bboxes, tr_velo_to_cam, r0_rect)
        
        for bbox_data in bboxes_lidar:
            x, y, z, l, h, w, yaw = bbox_data
            bbox = o3d.geometry.OrientedBoundingBox()
            bbox.center = [x, y, z + (h / 2.0)]
            bbox.extent = [w, l, h]
            R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, -yaw))
            bbox.R = R
            bbox.color = (1, 0, 0)
            bboxes.append(bbox)
        return bboxes

    def bbox_camera2lidar(self, bboxes, tr_velo_to_cam, r0_rect):
        x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
        xyz_size = np.concatenate([z_size, x_size, y_size], axis=1)
        extended_xyz = np.pad(bboxes[:, :3], ((0, 0), (0, 1)), 'constant', constant_values=1.0)
        rt_mat = np.linalg.inv(r0_rect @ tr_velo_to_cam)
        xyz = extended_xyz @ rt_mat.T
        bboxes_lidar = np.concatenate([xyz[:, :3], xyz_size, bboxes[:, 6:]], axis=1)
        return np.array(bboxes_lidar, dtype=np.float32)

    def key_callback_next(self, vis):
        self.current_idx = (self.current_idx + 1) % len(self.frames)
        self.load_frame(self.frames[self.current_idx])

    def key_callback_back(self, vis):
        self.current_idx = (self.current_idx - 1) % len(self.frames)
        self.load_frame(self.frames[self.current_idx])
        
    def key_callback_quit(self, vis):
        vis.destroy_window()

    def run(self):
        self.vis.register_key_callback(ord('N'), self.key_callback_next)
        self.vis.register_key_callback(ord('B'), self.key_callback_back)
        self.vis.register_key_callback(ord('Q'), self.key_callback_quit)
        self.vis.run()

def get_valid_frames(pcd_folder):
    files = [f for f in os.listdir(pcd_folder) if f.endswith(".bin")]
    frames = sorted([f.split('.')[0] for f in files])
    return frames

def main(args):
    pcd_folder = args.pcd_folder
    bbox_folder = args.bbox_folder
    calib_folder = args.calib_folder

    frames = get_valid_frames(pcd_folder)
    
    if args.idx == None: start_idx = frames[0] 
    else: start_idx = f"{args.idx:06}"
    
    no_bbox = args.no_bbox

    visualizer = Visualizer(pcd_folder, bbox_folder, calib_folder, frames, start_idx, no_bbox)
    visualizer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize point cloud with bounding boxes.")
    parser.add_argument('--pcd_folder', default='ds_pcd', help="Folder containing the point cloud files")
    parser.add_argument('--bbox_folder', default='submit', help="Folder containing the bounding box label files")
    parser.add_argument('--calib_folder', default='calib', help="Folder containing the calibration files")
    parser.add_argument('--idx', type=int, default=None, help="Starting frame index")
    parser.add_argument('--no_bbox', action='store_true', help='Disable bounding box visualization')
    args = parser.parse_args()
    main(args)

