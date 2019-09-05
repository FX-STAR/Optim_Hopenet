'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-09-05 12:09:33
@LastEditTime: 2019-09-05 15:49:17
@LastEditors: Please set LastEditors
'''
from mxnet.gluon import nn

__all__ = ["MobileFaceNet",
           "get_mobile_facenet"
           ]


def _make_conv(stage_index, channels=1, kernel=1, stride=1, pad=0,
               num_group=1, active=True):
    out = nn.HybridSequential(prefix='stage%d_' % stage_index)
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    out.add(nn.BatchNorm(scale=True))
    if active:
        out.add(nn.PReLU())
    return out


def _make_bottleneck(stage_index, layers, channels, stride, t, in_channels=0, use_se=False):
    layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
    with layer.name_scope():
        layer.add(Bottleneck(in_channels=in_channels, channels=channels, t=t, stride=stride, use_se=use_se))
        for _ in range(layers - 1):
            layer.add(Bottleneck(channels, channels, t, 1, use_se=use_se))
    return layer


class Bottleneck(nn.HybridBlock):

    def __init__(self, in_channels, channels, t, stride, use_se=False, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels == channels
        with self.name_scope():
            self.out = nn.HybridSequential()
            self.out.add(_make_conv(0, in_channels * t),
                         _make_conv(1, in_channels * t, kernel=3, stride=stride,
                                    pad=1, num_group=in_channels * t),
                         _make_conv(2, channels, active=False))
            # if use_se:
            #     self.out.add(SELayer(channels, channels))

    def hybrid_forward(self, F, x, **kwargs):
        out = self.out(x)
        if self.use_shortcut:
            out = F.elemwise_add(out, x)
        return out


class MobileFaceNet(nn.HybridBlock):
    """Mobile FaceNet"""
    def __init__(self, use_se=False, use_fc=True):
        super(MobileFaceNet, self).__init__(prefix='')
        with self.name_scope():
            self.features = nn.HybridSequential()
            self.features.add(_make_conv(0, 64, kernel=3, stride=2, pad=1),
                                _make_conv(0, 64, kernel=3, stride=1, pad=1, num_group=64))

            self.features.add(
                _make_bottleneck(1, layers=5, channels=64, stride=2, t=2, in_channels=64, use_se=use_se),
                _make_bottleneck(2, layers=1, channels=128, stride=2, t=4, in_channels=64, use_se=use_se),
                _make_bottleneck(3, layers=6, channels=128, stride=1, t=2, in_channels=128, use_se=use_se),
                _make_bottleneck(4, layers=1, channels=128, stride=2, t=4, in_channels=128, use_se=use_se),
                _make_bottleneck(5, layers=2, channels=128, stride=1, t=2, in_channels=128, use_se=use_se))

            self.features.add(_make_conv(6, 512),
                                _make_conv(6, 512, kernel=7, num_group=512, active=False),
                                # nn.Conv2D(128, 1, use_bias=False),
                                # nn.BatchNorm(scale=False, center=False),
                                # nn.Flatten()
                                )
        self.use_fc = use_fc
        self.fc_bin = nn.Dense(198, prefix='fc_bin_') # 66*3
        if self.use_fc:
            self.fc_pyr = nn.Dense(3, prefix='fc_pyr_') # pyr
            
    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.fc_bin(x)
        if self.use_fc:
            return x, self.fc_pyr(x)
        else:
            return x


def get_mobile_facenet(**kwargs):
    return MobileFaceNet(use_se=False, **kwargs)

# import mxnet as mx
# net = get_mobile_facenet()
# net.hybridize()
# x = mx.sym.var('data')
# y = net(x)
# y = mx.sym.Group(y) 
# mx.viz.print_summary(y, shape={"data": (1, 3, 112, 112)})
# graph=mx.viz.plot_network(y, title='./model_zoo/pose', node_attrs={"shape":"oval","fixedsize":"false"})
# graph.render()