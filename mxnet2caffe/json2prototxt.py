'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-07-27 16:20:17
@LastEditTime: 2019-09-02 19:57:12
@LastEditors: Please set LastEditors
'''
import argparse
import json
from prototxt_basic import write_node

def net_convert(mxnet_json, caffe_prototxt):
    with open(mxnet_json) as json_file:
        jdata = json.load(json_file)

    with open(caffe_prototxt, "w") as prototxt_file:
        nodes = jdata['nodes']
        for i, node in enumerate(nodes):
            if str(node['op']) == 'null' and str(node['name']) != 'data':
                continue

            info = {}
            if str(node['op']) == 'null' and str(node['name']) == 'data':
                info['op'] = 'Input'
            else:
                info['op'] = node['op']
            if 'attrs' in node:
                info['attrs'] = node['attrs']
            node['name'] = node['name'].replace('_fwd', '')  #### SymbolBlock的问题
            info['top'] = node['name']
            info['bottom'] = []
            info['params'] = []
            for input_ids in node['inputs']:
                input_node = nodes[input_ids[0]]
                if str(input_node['op']) != 'null' or (str(
                        input_node['name']) == 'data'):
                    info['bottom'].append(str(input_node['name']))

            write_node(prototxt_file, info)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MXNet jason to Caffe prototxt')
    parser.add_argument(
        '--mxnet-json', type=str, default='../../models/vanilla/r18/model-symbol.json')
    parser.add_argument(
        '--caffe-prototxt', type=str, default='test.prototxt')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    net_convert(args.mxnet_json, args.caffe_prototxt)
