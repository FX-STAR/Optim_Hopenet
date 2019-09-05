'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-06-22 22:02:36
@LastEditTime: 2019-08-31 17:44:56
@LastEditors: Please set LastEditors
'''
import os
import numpy as np
import cv2
import mxnet as mx
from mxnet.gluon.data.vision import transforms

class Dataset(mx.gluon.data.Dataset):
    # Head pose from 300W-LP or AFLW2000 dataset 
    def __init__(self, data_dir, file_path, transform=False):
        """ Args:
                data_dir: 300W_LP or AFLW2000 dir
                file_path: 300W_LP_pose.txt or AFLW2000_pose.txt
                transform: None
                k: expand ratio
        """
        self.data_dir = data_dir
        self.transform = transform
        self.lines = self._load_file(file_path)

    def __getitem__(self, index):
        line = self.lines[index].split()
        img_name = line[0]
        # radian to degree
        pyr=np.array([float(i)*180/np.pi for i in line[1:4]], dtype=np.float32)
        bbox=[int(i) for i in line[4:8]]
        img = cv2.imread(os.path.join(self.data_dir, img_name))
        if img is None:
            print(os.path.join(self.data_dir, img_name))

        h, w = img.shape[:2]
        # crop face
        k = np.random.random_sample() * 0.2 + 0.2  #(0.2-0.4)
        bb_w, bb_h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        x_min =np.clip(bbox[0]-0.6*k*abs(bb_w), 0, w)
        y_min =np.clip(bbox[1]-0.6*k*abs(bb_h), 0, h)
        x_max =np.clip(bbox[2]-0.6*k*abs(bb_w), 0, w)
        y_max =np.clip(bbox[3]-0.6*k*abs(bb_h), 0, h)
        img = img[int(y_min):int(y_max), int(x_min):int(x_max), :]

        if self.transform:
            img, pyr = self._transform(img, pyr)
        
        # roi_w, roi_h = x_max - x_min, y_max - y_min
        # roi = max(roi_w, roi_h)
        # ex_w, ex_h = int((roi-roi_w)//2), int((roi-roi_h)//2)
        # img = cv2.copyMakeBorder(img, ex_h, ex_h, ex_w, ex_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        h, w = img.shape[:2]
        max_=max(h,w)
        new=np.zeros((max_,max_,3), dtype=np.uint8)
        ex_h, ex_w = (max_-h)//2, (max_-w)//2
        new[ex_h:(ex_h+h),ex_w:(ex_w+w),:]=img
        
        pyr = np.clip(pyr, -99, 99)
        bin_label = np.digitize(pyr, range(-99,100,3))-1 
        bin_label = mx.nd.array(bin_label, dtype='float32')
        cont_label = mx.nd.array(pyr, dtype='float32')
        return self._preprocess(new), bin_label, cont_label


    def __len__(self):
        return len(self.lines)
    
    def _transform(self, img, pyr):
        # flip
        rnd = np.random.random_sample()
        if rnd < 0.5:
            pyr[1] *= -1
            pyr[2] *= -1
            img = cv2.flip(img, 1)
        return img, pyr
    
    def _preprocess(self, img):
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.transpose((2,0,1)).astype('float32')-127.5) *  0.0078125
        return mx.nd.array(img)
    
    def _load_file(self, file_path):
        with open(file_path) as f:
            lines = f.readlines()
        # first line is txt format
        return lines[1:]
