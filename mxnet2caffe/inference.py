'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-09-03 21:43:56
@LastEditTime: 2019-09-05 21:09:40
@LastEditors: Please set LastEditors
'''
import sys
caffe_python_root = '/home/lfx/Tool/caffe_cpu/python'
sys.path.append(caffe_python_root)
import caffe
import cv2
import numpy as np
from math import cos, sin

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

def crop(img, bbox):
    # crop face
    h, w = img.shape[:2]
    k = 0.3
    bb_w, bb_h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x_min =np.clip(bbox[0]-0.6*k*abs(bb_w), 0, w)
    y_min =np.clip(bbox[1]-0.6*k*abs(bb_h), 0, h)
    x_max =np.clip(bbox[2]+0.6*k*abs(bb_w), 0, w)
    y_max =np.clip(bbox[3]+0.6*k*abs(bb_h), 0, h)
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
    img = np.transpose(img, (2,0,1)).astype('float32')
    img = (img-127.5)/128
    return img

def get_caffe_out(net, data):
    net.blobs['data'].data[...] = data.copy()
    out = net.forward()
    pyr = out['fc_pyr']
    return pyr.copy()

if __name__ == "__main__": 
    caffe.set_mode_cpu()
    net = caffe.Net('./mxnet2caffe/model/caffe.prototxt', './mxnet2caffe/model/caffe.caffemodel', caffe.TEST)

    detect_type = 'mtcnn'

    img = cv2.imread('./test_res/test.jpg')
    if detect_type=='dlib':
        import dlib
        detector = dlib.get_frontal_face_detector()
        dets = detector(img, 0)
        bboxs = [(i.left(), i.top(), i.right(), i.bottom()) for i in dets]
    elif detect_type =='mtcnn':
        import sys
        sys.path.append('./')
        from mtcnn.mtcnn import MTCNN
        import mxnet as mx
        detector = MTCNN('./mtcnn/model/', ctx=mx.cpu(), num_worker=4, accurate_landmark=False)
        ret = detector.detect_face(img) 
        if ret is not None:
            bboxs, _ = ret
            bboxs = bboxs[:, :4]
    else:
        raise NotImplementedError
    
    if len(bboxs)>0:
        faces = np.array([crop(img, i) for i in bboxs])  
        pyrs = get_caffe_out(net, faces)
        print(pyrs)
        for pyr, (x1,y1,x2,y2) in zip(pyrs, bboxs):
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            img = draw_axis(img, pyr, tdx=(x1+x2)/2, tdy=(y1+y2)/2, size=100)
            img = cv2.rectangle(img, (x1,y1), (x2,y2), (0, 0, 255), 2)
            cv2.putText(img, 'pyr: (%.1f,%.1f,%.1f)'%(pyr[0], pyr[1], pyr[2]), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv2.imshow('demo', img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()    
    