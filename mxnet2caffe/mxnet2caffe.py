'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-09-01 22:24:31
@LastEditTime: 2019-09-05 11:06:56
@LastEditors: Please set LastEditors
'''
"""Adapted from https://github.com/cypw/MXNet2Caffe"""

import os, sys
import argparse
import mxnet as mx
from find_caffe import caffe
import caffe
import os.path as osp
from json2prototxt import net_convert


def weight_convert(mxnet_prefix, mxnet_epoch, caffe_prototxt, caffe_model):
    # load mxnet model
    _, arg_params, aux_params = mx.model.load_checkpoint(
        mxnet_prefix, mxnet_epoch)
    # load caffe net define
    net = caffe.Net(caffe_prototxt, caffe.TEST)

    # convert weight
    all_keys = list(arg_params.keys()) + list(aux_params.keys())
    all_keys.sort()
   
    for i, key in enumerate(all_keys):
        try:
            if 'data' is key:
                pass
            elif '_weight' in key:
                ckey = key.replace('_weight', '')
                net.params[ckey][0].data.flat = arg_params[key].asnumpy().flat
            elif '_bias' in key:
                ckey = key.replace('_bias', '')
                net.params[ckey][1].data.flat = arg_params[key].asnumpy().flat
            elif '_gamma' in key and 'relu' not in key:
                ckey = key.replace('_gamma', '_scale')
                net.params[ckey][0].data.flat = arg_params[key].asnumpy().flat
            elif '_gamma' in key and 'relu' in key:  # for prelu
                ckey = key.replace('_gamma', '')
                assert (len(net.params[ckey]) == 1)
                net.params[ckey][0].data.flat = arg_params[key].asnumpy().flat
            elif '_alpha' in key: # prelu
                ckey = key.replace('_alpha', '')
                net.params[ckey][0].data.flat = arg_params[key].asnumpy().flat
            elif '_beta' in key:
                ckey = key.replace('_beta', '_scale')
                net.params[ckey][1].data.flat = arg_params[key].asnumpy().flat
            # elif '_moving_mean' in key:
            #     ckey = key.replace('_moving_mean', '')
            #     net.params[ckey][0].data.flat = aux_params[key].asnumpy().flat
            #     net.params[ckey][2].data[...] = 1
            # elif '_moving_var' in key:
            #     ckey = key.replace('_moving_var', '')
            #     net.params[ckey][1].data.flat = aux_params[key].asnumpy().flat
            #     net.params[ckey][2].data[...] = 1
            elif '_running_mean' in key:
                ckey = key.replace('_running_mean', '')
                net.params[ckey][0].data.flat = aux_params[key].asnumpy().flat
                net.params[ckey][2].data[...] = 1
            elif '_running_var' in key:
                ckey = key.replace('_running_var', '')
                net.params[ckey][1].data.flat = aux_params[key].asnumpy().flat
                net.params[ckey][2].data[...] = 1
            else:
                sys.exit("Warning!  Unknown mxnet: {}".format(key))

            # print("% 3d | %s -> %s" % (i, key.ljust(40), ckey.ljust(30)))

        except KeyError:
            if key != 'fc7_weight':
                print("\nWarning!  key error mxnet: {}".format(key))

    net.save(caffe_model)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MXNet model to Caffe model')
    parser.add_argument('--save', type=str, default='./model', help='save path')
    parser.add_argument('--prefix', type=str, default='best_pose', help='mxnet prefix')
    parser.add_argument('--epoch', type=int, default=0, help='mxnet epoch')
    parser.add_argument('--prototxt', type=str, default='caffe.prototxt')
    parser.add_argument('--caffemodel', type=str, default='caffe.caffemodel')
    parser.add_argument('--trans', type=str, default='net', help='net or weight')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    assert args.trans in ('net', 'weight')
    prototxt = osp.join(args.save, args.prototxt)
    caffemodel = osp.join(args.save, args.caffemodel)
    if args.trans == 'net':
        json = osp.join(args.save, args.prefix + '-symbol.json')
        net_convert(json, prototxt)
        print('Convert net define from %s to %s' % (json, prototxt))
    elif args.trans == 'weight':
        params = osp.join(args.save, args.prefix)
        weight_convert(params, args.epoch, prototxt, caffemodel)
        print('Convert weight define from %s to %s' % (params, caffemodel))

    
