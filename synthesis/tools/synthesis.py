#!/usr/bin/env python

import h5py
import json
import argparse
import pprint
import time, os, sys
import numpy as np
import os.path as osp
import threading
import libsynthesizer
from transforms3d.quaternions import quat2mat


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Synthesis of linemod')
    parser.add_argument('--points_path', dest='points_path',
                        help='Path to the xyz format file representing the points of the object model',
                        default=None, type=str, required=True)
    parser.add_argument('--model_path', dest='model_path',
                        help='Path to the obj format file representing the object model',
                        default=None, type=str, required=True)
    parser.add_argument('--pose_path', dest='pose_path',
                        help='Path to the poses file',
                        default=None, type=str, required=True)
    parser.add_argument('--extent_path', dest='extent_path',
                        help='Path to the extent file',
                        default=None, type=str, required=True)
    parser.add_argument('--nb_img', dest='nb_img',
                        help='Number of images to generate',
                        default=None, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='Directory where the synthesized images are saved',
                        default=None, type=str)
    parser.add_argument('--width', dest='width',
                        help='Width of the synthesized images',
                        default=640, type=int)
    parser.add_argument('--height', dest='height',
                        help='Height of the synthesized images',
                        default=480, type=int)

    

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def render_one(intrinsic_matrix, extents, points, nb_img):
    synthesizer = libsynthesizer.Synthesizer(args.model_path, args.pose_path)
    synthesizer.setup(args.width, args.height)

    which_class = 0
    width = args.width
    height = args.height

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    zfar = 6.0
    znear = 0.25
    factor_depth = 1000.0
    dataset = dict()
    print('Getting in the while loop')
    n = 0
    while n < nb_img:
        print(n)

        # render a synthetic image
        im_syn = np.zeros((height, width, 4), dtype=np.float32)
        depth_syn = np.zeros((height, width, 3), dtype=np.float32)
        vertmap_syn = np.zeros((height, width, 3), dtype=np.float32)
        poses = np.zeros((1, 7), dtype=np.float32)
        centers = np.zeros((1, 2), dtype=np.float32)

        synthesizer.render_one_python(int(which_class), int(width), int(height), fx, fy, px, py, znear, zfar, \
            im_syn, depth_syn, vertmap_syn, poses, centers, extents)

        # convert images
        im_syn = np.clip(255 * im_syn, 0, 255)
        im_syn = im_syn.astype(np.uint8)
        depth_syn = depth_syn[:, :, 0]

        # convert depth
        im_depth_raw = factor_depth * 2 * zfar * znear / (zfar + znear - (zfar - znear) * (2 * depth_syn - 1))
        I = np.where(depth_syn == 1)
        im_depth_raw[I[0], I[1]] = 0

        # compute labels from vertmap
        label = np.round(vertmap_syn[:, :, 0]) + 1
        label[np.isnan(label)] = 0

        I = np.where(label != which_class + 1)
        label[I[0], I[1]] = 0

        I = np.where(label == which_class + 1)
        if len(I[0]) < 800:
            continue

        # convert pose
        qt = np.zeros((3, 4, 1), dtype=np.float32)
        qt[:, :3, 0] = quat2mat(poses[0, :4])
        qt[:, 3, 0] = poses[0, 4:]

        # process the vertmap
        vertmap_syn[:, :, 0] = vertmap_syn[:, :, 0] - np.round(vertmap_syn[:, :, 0])
        vertmap_syn[np.isnan(vertmap_syn)] = 0

        # compute box
        x3d = np.ones((4, points.shape[1]), dtype=np.float32)
        cls = 1
        x3d[0, :] = points[cls,:,0]
        x3d[1, :] = points[cls,:,1]
        x3d[2, :] = points[cls,:,2]
        RT = qt[:, :, 0]
        x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
        x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
        x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
        box = np.zeros((1, 4), dtype=np.float32)
        box[0, 0] = np.min(x2d[0, :])
        box[0, 1] = np.min(x2d[1, :])
        box[0, 2] = np.max(x2d[0, :])
        box[0, 3] = np.max(x2d[1, :])

        # metadata
        metadata = {'poses': qt, 'center': centers, 'box': box, \
                    'cls_indexes': np.array([which_class + 1]), 'intrinsic_matrix': intrinsic_matrix, 'factor_depth': factor_depth}

        # construct data
        data = {'image': im_syn, 'depth': im_depth_raw.astype(np.uint16), 'label': label.astype(np.uint8), 'meta_data': metadata}

        dataset[n] = data
        n += 1
    
    output_path = os.path.abspath(os.path.join(args.output_dir, 'dataset.h5'))
    with h5py.File(output_path, 'w') as hdf:
        hdf.create_dataset('data', data=json.dumps(dataset, cls=NumpyEncoder))
    print('Sucessfully saved the synthesized dataset to: ', output_path)


# In the sequel 2 is the number of classes: background and object
def load_object_points(point_file):

    points = [[] for _ in range(2)]
    num = np.inf

    for i in range(1, 2):
        assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
        points[i] = np.loadtxt(point_file)
        if points[i].shape[0] < num:
            num = points[i].shape[0]

    points_all = np.zeros((2, num, 3), dtype=np.float32)

    for i in range(1, 2):
        points_all[i, :, :] = points[i][:num, :]

    return points[1], points_all

def load_object_extents(extent_file):
        assert os.path.exists(extent_file), \
                'Path does not exist: {}'.format(extent_file)
        extents = np.zeros((2, 3), dtype=np.float32)
        extents_all = np.loadtxt(extent_file)
        print('extents_all loaded', extents_all)
        extents[1, :] = extents_all

        return extents

if __name__ == '__main__':
    args = parse_args()
    _, _points_all = load_object_points(args.points_path)
    _extents_all = load_object_extents(args.extent_path)
    fx = 572.41140
    fy = 573.57043
    px = 325.26110
    py = 242.04899
    intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])

    t = threading.Thread(target=render_one, args=(intrinsic_matrix, _extents_all, _points_all, int(args.nb_img)))
    t.start()
