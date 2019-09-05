'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-28 15:11:40
@LastEditTime: 2019-08-31 16:49:42
@LastEditors: Please set LastEditors
'''
import cv2
import numpy as np
import math
from math import cos, sin
import os

def draw_axis(img, pitch, yaw, roll, tdx=None, tdy=None, size = 100):

    # pitch = pitch * np.pi / 180
    # yaw = -(yaw * np.pi / 180)
    # roll = roll * np.pi / 180
    print(pitch*180/np.pi, yaw*180/np.pi, roll*180/np.pi)
    yaw*=-1

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img


# data_root = '/home/lfx/Data/300W_LP'
# ann_txt = './data/300W_LP_pose.txt'
data_root = '/home/lfx/Data/AFLW2000'
ann_txt = './data/AFLW2000_pose.txt' 
with open(ann_txt, 'r') as f:
    lines = f.readlines()
    for l in lines[1:20]:
        print(l.strip())
        l = l.split()
        im_name = l[0]
        pyr = l[1:4]
        bbox = l[4:]
        img = cv2.imread(os.path.join(data_root, im_name))
        img = draw_axis(img, float(pyr[0]), float(pyr[1]), float(pyr[2]), tdx = (int(bbox[0])+int(bbox[2]))/2, tdy= (int(bbox[1])+int(bbox[3]))/2, size=100)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255))
        cv2.imwrite('data/test/'+os.path.basename(im_name), img)

        
# # discard large pose
# ann_txt = './data/AFLW2000_pose.txt' 
# save_txt = './data/AFLW2000_pose_save.txt'
# dis_txt = './data/AFLW2000_pose_dis.txt'
# f_save = open(save_txt, 'w')
# f_dis = open(dis_txt, 'w')
# with open(ann_txt, 'r') as f:
#     lines = f.readlines()
#     f_save.write('image_name, pitch, yaw, roll(euler), xmin, ymin, xmax, ymax \n')
#     f_dis.write('image_name, pitch, yaw, roll(euler), xmin, ymin, xmax, ymax \n')
#     for l in lines[1:]:
#         ls = l.split()
#         pyr = ls[1:4]
#         degree = abs(max([float(i) for i in pyr])*180/np.pi)
#         if degree>99:
#             f_dis.write(l)
#         else:
#             f_save.write(l)
# f_save.close()
# f_dis.close()
            



        