'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-07-27 15:23:30
@LastEditTime: 2019-09-03 21:47:26
@LastEditors: Please set LastEditors
'''
# prototxt_basic


def Input(txt_file, info):
    txt_file.write('name: "mxnet-mdoel"\n')
    txt_file.write('layer {\n')
    txt_file.write('  name: "data"\n')
    txt_file.write('  type: "Input"\n')
    txt_file.write('  top: "data"\n')
    txt_file.write('  input_param {\n')
    txt_file.write('    shape: { dim: 1 dim: 3 dim: 112 dim: 112 }\n')  # TODO
    txt_file.write('  }\n')
    txt_file.write('}\n')
    txt_file.write('\n')


def Convolution(txt_file, info):
    if 'no_bias' in info['attrs'] and info['attrs']['no_bias'] == 'True':
        bias_term = 'false'
    else:
        bias_term = 'true'

    txt_file.write('layer {\n')
    if info['top'] == 'conv0':
        txt_file.write('	bottom: "data"\n')
    else:
        txt_file.write('	bottom: "%s"\n' % info['bottom'][0])
    txt_file.write('	top: "%s"\n' % info['top'])
    txt_file.write('	name: "%s"\n' % info['top'])
    txt_file.write('	type: "Convolution"\n')
    txt_file.write('	convolution_param {\n')
    txt_file.write('		num_output: %s\n' % info['attrs']['num_filter'])
    txt_file.write('		kernel_size: %s\n' %
                   info['attrs']['kernel'].split('(')[1].split(',')[0])  # TODO
    if 'pad' in info['attrs']:
        txt_file.write(
            '		pad: %s\n' %
            info['attrs']['pad'].split('(')[1].split(',')[0])  # TODO
    if 'num_group' in info['attrs']:
        txt_file.write('		group: %s\n' % info['attrs']['num_group'])
        # dw 卷积caffe支持问题
        if int(info['attrs']['num_group'])>1:
            txt_file.write('		engine: CAFFE\n')

    if 'stride' in info['attrs']:
        txt_file.write('		stride: %s\n' %
                       info['attrs']['stride'].split('(')[1].split(',')[0])
    txt_file.write('		bias_term: %s\n' % bias_term)
    txt_file.write('	}\n')
    txt_file.write('}\n')
    txt_file.write('\n')


def BatchNorm(txt_file, info):
    txt_file.write('layer {\n')
    txt_file.write('  bottom: "%s"\n' % info['bottom'][0])
    txt_file.write('  top: "%s"\n' % info['top'])
    txt_file.write('  name: "%s"\n' % info['top'])
    txt_file.write('  type: "BatchNorm"\n')
    txt_file.write('  batch_norm_param {\n')
    txt_file.write('    use_global_stats: true\n')  # TODO
    if 'momentum' in info['attrs']:
        txt_file.write(
            '    moving_average_fraction: %s\n' % info['attrs']['momentum'])
    else:
        txt_file.write('    moving_average_fraction: 0.9\n')
    if 'eps' in info['attrs']:
        txt_file.write('    eps: %s\n' % info['attrs']['eps'])
    else:
        txt_file.write('    eps: 0.001\n')
    txt_file.write('  }\n')
    txt_file.write('}\n')
    txt_file.write('layer {\n')
    txt_file.write('  bottom: "%s"\n' % info['top'])
    txt_file.write('  top: "%s"\n' % info['top'])
    txt_file.write('  name: "%s_scale"\n' % info['top'])
    txt_file.write('  type: "Scale"\n')
    txt_file.write('  scale_param { bias_term: true }\n')
    txt_file.write('}\n')
    txt_file.write('\n')

## Relu
def Activation(txt_file, info):  
    if info['top'].startswith('activation'):
        return
    txt_file.write('layer {\n')
    txt_file.write('  bottom: "%s"\n' % info['bottom'][0])
    txt_file.write('  top: "%s"\n' % info['top'])
    txt_file.write('  name: "%s"\n' % info['top'])
    txt_file.write('  type: "ReLU"\n')  # TODO
    txt_file.write('}\n')
    txt_file.write('\n')

def LeakyReLU(txt_file, info):
    if info['attrs']['act_type'] == 'elu':
        txt_file.write('layer {\n')
        txt_file.write('  bottom: "%s"\n' % info['bottom'][0])
        txt_file.write('  top: "%s"\n' % info['top'])
        txt_file.write('  name: "%s"\n' % info['top'])
        txt_file.write('  type: "ELU"\n')
        txt_file.write('  elu_param { alpha: 0.25 }\n')
        txt_file.write('}\n')
        txt_file.write('\n')
    elif info['attrs']['act_type'] == 'prelu':
        txt_file.write('layer {\n')
        txt_file.write('  bottom: "%s"\n' % info['bottom'][0])
        txt_file.write('  top: "%s"\n' % info['top'])
        txt_file.write('  name: "%s"\n' % info['top'])
        txt_file.write('  type: "PReLU"\n')
        txt_file.write('}\n')
        txt_file.write('\n')
    else:
        raise Exception("unsupported Activation")
        
def Relu6(txt_file, info):
    txt_file.write('layer {\n')
    txt_file.write('  bottom: "%s"\n' % info['bottom'][0])
    txt_file.write('  top: "%s"\n' % info['top'])
    txt_file.write('  name: "%s"\n' % info['top'])
    txt_file.write('  type: "ReLU6"\n')  # TODO
    txt_file.write('}\n')
    txt_file.write('\n')

def Concat(txt_file, info):
    txt_file.write('layer {\n')
    txt_file.write('  name: "%s"\n' % info['top'])
    txt_file.write('  type: "Concat"\n')
    for bottom_i in info['bottom']:
        txt_file.write('  bottom: "%s"\n' % bottom_i)
    txt_file.write('  top: "%s"\n' % info['top'])
    txt_file.write('}\n')
    txt_file.write('\n')


def Pooling(txt_file, info):
    pool_type = 'AVE' if info['attrs']['pool_type'] == 'avg' else 'MAX'
    txt_file.write('layer {\n')
    txt_file.write('  bottom: "%s"\n' % info['bottom'][0])
    txt_file.write('  top: "%s"\n' % info['top'])
    txt_file.write('  name: "%s"\n' % info['top'])
    txt_file.write('  type: "Pooling"\n')
    txt_file.write('  pooling_param {\n')
    txt_file.write('    pool: %s\n' % pool_type)  # TODO
    if pool_type!='AVE':
        txt_file.write('    kernel_size: %s\n' % info['attrs']['kernel'].split('(')[1].split(',')[0])
        txt_file.write('    stride: %s\n' % info['attrs']['stride'].split('(')[1].split(',')[0])
        if 'pad' in info['attrs']:
            txt_file.write('    pad: %s\n' % info['attrs']['pad'].split('(')[1].split(',')[0])
    else:
        txt_file.write('    engine: CAFFE\n')
        txt_file.write('    global_pooling: true\n')
    txt_file.write('  }\n')
    txt_file.write('}\n')
    txt_file.write('\n')


def Flatten(txt_file, info):
    txt_file.write('layer {\n')
    txt_file.write('  name: "%s"\n' % info['top'])
    txt_file.write('  type: "Flatten"\n')
    for bottom_i in info['bottom']:
        txt_file.write('  bottom: "%s"\n' % bottom_i)
    txt_file.write('  top: "%s"\n' % info['top'])
    txt_file.write('}\n')
    txt_file.write('\n')


def SoftmaxOutput(txt_file, info):
    raise NotImplementedError


def Normalize(txt_file, info):
    txt_file.write('layer {\n')
    txt_file.write('  bottom: "%s"\n' % info['bottom'][0])
    txt_file.write('  top: "%s"\n' % info['top'])
    txt_file.write('  name: "%s"\n' % info['top'])
    txt_file.write('  type: "Normalize"\n')
    txt_file.write('  norm_param {\n')
    txt_file.write('    scale_filler {\n')
    txt_file.write('      type: "constant"\n')
    txt_file.write('      value: 1\n')
    txt_file.write('    }\n')
    txt_file.write('  }\n')
    txt_file.write('}\n')
    txt_file.write('\n')

def Eltwise(txt_file, info, op):
    txt_file.write('layer {\n')
    txt_file.write('  type: "Eltwise"\n')
    txt_file.write('  top: "%s"\n' % info['top'])
    txt_file.write('  name: "%s"\n' % info['top'])
    for btom in info['bottom']:
        txt_file.write('  bottom: "%s"\n' % btom)
    txt_file.write('  eltwise_param { operation: %s }\n' % op)
    txt_file.write('}\n')
    txt_file.write('\n')

def ElementWiseSum(txt_file, info):
    if info['bottom'][1].startswith('broadcast'):
        return

    txt_file.write('layer {\n')
    txt_file.write('  name: "%s"\n' % info['top'])
    txt_file.write('  type: "Eltwise"\n')
    for bottom_i in info['bottom']:
        txt_file.write('  bottom: "%s"\n' % bottom_i)
    txt_file.write('  top: "%s"\n' % info['top'])
    txt_file.write('  eltwise_param { operation: SUM }\n')
    txt_file.write('}\n')
    txt_file.write('\n')

def ElementWiseProd(txt_file, info):
    if info['bottom'][1].startswith('broadcast'):
        return
    txt_file.write('layer {\n')
    txt_file.write('  name: "%s"\n' % info['top'])
    txt_file.write('  type: "Eltwise"\n')
    for bottom_i in info['bottom']:
        txt_file.write('  bottom: "%s"\n' % bottom_i)
    txt_file.write('  top: "%s"\n' % info['top'])
    txt_file.write('  eltwise_param { operation: PROD }\n')
    txt_file.write('}\n')
    txt_file.write('\n')

def Power(txt_file, info):
    power, scale, shift = 1, 1 , 0
    if 'plus' in info['op']:
        shift = float(info['attrs']['scalar'])
        out = '    shift:%f\n'%shift
    if 'div' in info['op']:
        scale = 1/float(info['attrs']['scalar'])
        out = '    scale:%f\n'%scale
    if 'mul' in info['op']:
        scale = float(info['attrs']['scalar'])
        out = '    scale:%f\n'%scale
    txt_file.write('layer {\n')
    txt_file.write('  name: "%s"\n' % info['top'])
    txt_file.write('  type: "Power"\n')
    for bottom_i in info['bottom']:
        txt_file.write('  bottom: "%s"\n' % bottom_i)
    txt_file.write('  top: "%s"\n' % info['top'])
    txt_file.write('  power_param {\n')
    txt_file.write(out)
    txt_file.write('  }\n')
    txt_file.write('}\n')


def FullyConnected(txt_file, info):
    if info['bottom'][0].startswith('_mulscalar'):
        return

    txt_file.write('layer {\n')
    if info['bottom'][0] == 'dropout0':
        txt_file.write('  bottom: "hardswish18__mul0"\n')
    else:
        txt_file.write('  bottom: "%s"\n' % info['bottom'][0])
    txt_file.write('  top: "%s"\n' % info['top'])
    txt_file.write('  name: "%s"\n' % info['top'])
    txt_file.write('  type: "InnerProduct"\n')
    txt_file.write('  inner_product_param {\n')
    txt_file.write('    num_output: %s\n' % info['attrs']['num_hidden'])
    txt_file.write('  }\n')
    txt_file.write('}\n')
    txt_file.write('\n')

def Broadcastmul(txt_file, info):
    txt_file.write('layer {\n')
    txt_file.write('  name: "%s"\n' % info['top'])
    txt_file.write('  type: "Broadcastmul"\n')
    for bottom_i in info['bottom']:
        txt_file.write('  bottom: "%s"\n' % bottom_i)
    txt_file.write('  top: "%s"\n' % info['top'])
    txt_file.write('}\n')
    txt_file.write('\n')

# ----------------------------------------------------------------
def write_node(txt_file, info):
    if 'label' in info['top']:
        return
    if info['op'] == 'Input':
        Input(txt_file, info)
    elif info['op'] in ('Convolution', 'ChannelwiseConvolution'):
        Convolution(txt_file, info)
    elif info['op'] == 'BatchNorm':
        BatchNorm(txt_file, info)
    elif info['op'] == 'Activation':
        Activation(txt_file, info)
    elif info['op'] == 'Concat':
        Concat(txt_file, info)
    elif info['op'] == 'Pooling':
        Pooling(txt_file, info)
    elif info['op'] == 'Flatten':
        Flatten(txt_file, info)
    elif info['op'] == 'FullyConnected':
        FullyConnected(txt_file, info)
    elif info['op'] == 'SoftmaxOutput':
        SoftmaxOutput(txt_file, info)
    elif info['op'] == 'LeakyReLU':
        LeakyReLU(txt_file, info)
    elif info['op'] == 'L2Normalization':
        Normalize(txt_file, info)
    elif info['op'] in ('ElementWiseSum', '_Plus', 'elemwise_add'):
        Eltwise(txt_file, info, 'SUM')
    elif info['op'] == 'elemwise_mul':
        Eltwise(txt_file, info, 'PROD')
    elif info['op'] == 'broadcast_mul':
        Broadcastmul(txt_file, info)
    elif '_scalar' in info['op']: # _plus_scalar, _div_scalar,_mul_scalar  
        Power(txt_file, info)
    elif info['op'] == 'clip' and info['attrs']['a_max']=='6' and info['attrs']['a_min']=='0':
        Relu6(txt_file, info)
    elif info['op'] == 'Dropout':
        # Need manual convert prototxt
        # Todo
        pass
    else:
        assert 0, "Warning! Skip Unknown mxnet op:{}".format(info['op'])
