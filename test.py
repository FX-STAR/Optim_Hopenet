'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-30 14:38:32
@LastEditTime: 2019-09-05 10:40:57
@LastEditors: Please set LastEditors
'''


import mxnet as mx
import cv2
import numpy as np
from mxnet.gluon import nn as gnn
from math import cos, sin
import argparse
import os
import os.path as osp
from mtcnn.mtcnn import MTCNN

def draw_axis(img, pyr, tdx=None, tdy=None, size = 100):
    pitch = pyr[0] * np.pi / 180
    yaw = -(pyr[1] * np.pi / 180)
    roll = pyr[2] * np.pi / 180

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


def get_net(_ctx, json, params):
    inputs = mx.sym.var('data', dtype='float32')
    net = gnn.SymbolBlock(mx.sym.load(json)['fc_pyr_fwd_output'], inputs)
    net.load_parameters(params, ctx=_ctx)
    net.hybridize()
    return net

def crop(img, bbox):
    # crop face
    h, w = img.shape[:2]
    k = 0.3
    bb_w, bb_h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x_min =np.clip(bbox[0]-0.6*k*abs(bb_w), 0, w)
    y_min =np.clip(bbox[1]-0.6*k*abs(bb_h), 0, h)
    x_max =np.clip(bbox[2]-0.6*k*abs(bb_w), 0, w)
    y_max =np.clip(bbox[3]-0.6*k*abs(bb_h), 0, h)
    img = img[int(y_min):int(y_max), int(x_min):int(x_max), :]
    roi_w = x_max - x_min
    roi_h = y_max - y_min
    roi = max(roi_w, roi_h)
    ex_w = int((roi-roi_w)//2)
    ex_h = int((roi-roi_h)//2)
    img = cv2.copyMakeBorder(img, ex_h, ex_h, ex_w, ex_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # prercocess
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.transpose((2,0,1)).astype('float32')-127.5) *  0.0078125
    return img


def predict_image(img, detector, json, params, _ctx):
    if args.detector == 'dlib':
        dets = detector(img, 0)
        bboxs = [(i.left(), i.top(), i.right(), i.bottom()) for i in dets]
        if len(bboxs)==0:
            return img
    else:
        ret = detector.detect_face(img) 
        if ret is None:
            return img
        bboxs, _ = ret
        bboxs = bboxs[:, :4]
    faces = [crop(img, i) for i in bboxs]
    faces = mx.nd.array(faces, _ctx)
    net = get_net(_ctx, json, params)
    pyrs=net(faces).asnumpy()
    for pyr, (x1,y1,x2,y2) in zip(pyrs, bboxs):
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
        img = draw_axis(img, pyr, tdx=(x1+x2)/2, tdy=(y1+y2)/2, size=100)
        img = cv2.rectangle(img, (x1,y1), (x2,y2), (0, 0, 255), 2)
        cv2.putText(img, 'pyr: (%.1f,%.1f,%.1f)'%(pyr[0], pyr[1], pyr[2]), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    return img


def get_args():
    parser = argparse.ArgumentParser(description='Test config.')
    parser.add_argument('--test_type', type=str, default='image', help='image, video, camera')
    parser.add_argument('--image', type=str, default='./test_res/test.jpg', help='test image path')
    parser.add_argument('--video', type=str, default='./test_res/test.mp4', help='test video path')
    parser.add_argument('--save', type=str, default='./test_res', help='result save path')

    parser.add_argument('--use_gpu', type=int, default=0)
    parser.add_argument('--detector', type=str, default='mtcnn', help='mtcnn, dlib')
    # mxnet 
    parser.add_argument('--json', type=str, default='./weight/v3_small_alpha1/best_pose-symbol.json')
    parser.add_argument('--params', type=str, default='./weight/v3_small_alpha1/best_pose-0000.params')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    _ctx=mx.gpu(0) if args.use_gpu else mx.cpu()
    json, params = args.json, args.params
    
    assert args.detector in ('mtcnn', 'dlib')
    if args.detector=='mtcnn':
        detector = MTCNN('./mtcnn/model/', ctx=_ctx, num_worker=4, accurate_landmark=False)
    elif args.detector=='dlib':
        import dlib
        detector = dlib.get_frontal_face_detector()

    if args.test_type == 'image':
        image = cv2.imread(args.image)
        image = predict_image(image, detector, json, params, _ctx)
        cv2.imwrite(osp.join(args.save, osp.basename(args.image).replace('.', '_pre.')), image)
        cv2.imshow('demo', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
    
    elif args.test_type == 'video':
        cap = cv2.VideoCapture(args.video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out = cv2.VideoWriter(osp.join(args.save, osp.basename(args.video).split('.')[0]+'_pre.mp4'), fourcc, fps, (width, height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = predict_image(frame, detector, json, params, _ctx)
            #out.write(frame)
            cv2.imshow("demo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        #out.release()
        cv2.destroyAllWindows()
    
    elif args.test_type == 'camera':
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            frame = predict_image(frame, detector, json, params, _ctx)
            cv2.imshow("demo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        raise NotImplementedError


