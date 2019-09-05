'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-28 15:06:12
@LastEditTime: 2019-08-31 17:19:43
@LastEditors: Please set LastEditors
'''
import scipy.io as sio
import os
import argparse
import cv2
import numpy as np

def get_pyr_from_mat(mat):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_bbox(size, mat):
    # Get 2D landmarks
    w, h = size
    pt2d = mat['pt2d']
    x, y = pt2d
    x_min = min(x[(x>=0)&(x<=w)])
    y_min = min(y[(y>=0)&(y<=h)])
    x_max = max(x[(x>=0)&(x<=w)])
    y_max = max(y[(y>=0)&(y<=h)])
    return [x_min, y_min, x_max, y_max]

def get_args():
    parser = argparse.ArgumentParser(description='Train head pose by mobilenetv3.')
    # dataset
    parser.add_argument('--data_root', type=str, default='/home/lfx/Data')
    parser.add_argument('--dataset', type=str, default='300W_LP, AFLW2000')
    parser.add_argument('--save_path', type=str, default='./data')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    data_root = args.data_root
    save_path = args.save_path
    dataset = args.dataset.strip().split(',')
    for data in dataset:
        data = data.strip()
        print('Parse dataset: %s'%data)
        if '300W_LP' in data:
            dataset_list = ['AFW', 'AFW_Flip', 'HELEN', 'HELEN_Flip', 'IBUG', 'IBUG_Flip',  'LFPW', 'LFPW_Flip']
        elif 'AFLW2000' in data:
            dataset_list=['']
        else:
            raise NotImplementedError
        f = open(os.path.join(save_path, '%s_pose1.txt'%data), 'w')
        f.write('image_name, pitch, yaw, roll(euler), xmin, ymin, xmax, ymax \n')
        for ds in dataset_list:
            _set = os.path.join(data_root, data, ds)
            print(_set)
            files_list = os.listdir(_set)
            for _file in files_list:
                if _file[-3:]!='mat':
                    continue
                jpg = os.path.join(ds, _file[:-4]+'.jpg')
                mat = sio.loadmat(os.path.join(_set, _file))
                pyr = get_pyr_from_mat(mat)
                size = cv2.imread(os.path.join(_set, _file[:-4]+'.jpg')).shape[:2]
                bbox = get_bbox(size, mat)
                jpg_anno=[]
                jpg_anno.append(jpg)
                if abs(max(pyr)*180/np.pi)>99:
                    continue
                for i in pyr:
                    jpg_anno.append(str(i))
                for i in bbox:
                    jpg_anno.append(str(int(round(i))))
                f.write('\t'.join(jpg_anno)+'\n')
        f.close()



