'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-06-22 23:02:37
@LastEditTime: 2019-09-05 15:40:59
@LastEditors: Please set LastEditors
'''
import mxnet as mx
import mxnet.gluon as nn
from model_zoo.mobilenetv3 import get_mobilenet_v3
from model_zoo.mobilenetv2 import get_mobilenet_v2
from model_zoo.mobilefacenet import get_mobile_facenet
from dataset import Dataset
import argparse
import numpy as np
import os
import os.path as osp
import time
from gluoncv.utils import TrainingHistory


def get_args():
    parser = argparse.ArgumentParser(description='Train head pose by mobilenetv3.')
    # dataset
    parser.add_argument('--dataset', type=str, default='/home/lfx/Data/300W_LP')
    parser.add_argument('--anno_txt', type=str, default='./data/300W_LP_pose.txt')
    parser.add_argument('--num_workers', type=int, default=6, help='io workers')
    # train
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_type', type=str, default='cos')
    parser.add_argument('--lr_decay_epoch', type=str, default='10, 15, 20')
    parser.add_argument('--wd', type=float, default=4e-5, help='weight decay') 
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.01)

    # net
    parser.add_argument('--version', type=str, default='small')
    parser.add_argument('--width_mult', type=float, default=1)
    parser.add_argument('--use_fc', type=int, default=0)
    parser.add_argument('--net', type=str, default='v3')
    
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--log_interval', type=int,default=100)
    parser.add_argument('--save', type=str, default='./weight', help='save model path')
    parser.add_argument('--prefix', type=str, default='test', help='save model path prefix')
    args = parser.parse_args()
    return args

def get_net(ctx, args):
    json, param=None, None
    if args.weights:
        json, params = [i.strip() for i in args.weights.strip().split()]
    assert args.width_mult<=1 and args.width_mult>0
    assert args.version in ('small', 'large')
    if args.net=='v3':
        net = get_mobilenet_v3(args.version, multiplier=args.width_mult, use_fc=args.use_fc)
    elif args.net=='v2':
        net = get_mobilenet_v2(multiplier=args.width_mult, use_fc=args.use_fc)
    elif args.net == 'facenet':
        net = get_mobile_facenet(use_fc=args.use_fc)
    net.initialize(init=mx.init.Xavier(), ctx=ctx)
    net.hybridize()
    return net

def get_data(args):
    """
    Returns:
        train_loader: train datset loader
        val_loader: val list datset loader
    """
    train_ = Dataset('/home/lfx/Data/300W_LP', './data/300W_LP_pose.txt', transform=True)
    train_loader =  mx.gluon.data.DataLoader(train_, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, last_batch='rollover')
 
    val_ = Dataset('/home/lfx/Data/AFLW2000', './data/AFLW2000_pose.txt', transform=False)
    val_loader =  mx.gluon.data.DataLoader(val_, batch_size=args.bs, num_workers=args.num_workers, last_batch='keep')
    return train_loader, val_loader


def cal_loss(outputs, bin_label, cont_label, _ctx, args):
    # loss
    ce_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    mse_loss = nn.loss.L2Loss()
    
    if args.use_fc:
        outputs_bin, pyr = outputs
        pitch, yaw, roll = outputs_bin[:, :66], outputs_bin[:, 66:66*2], outputs_bin[:, 66*2:]
        ce_pitch = ce_loss(pitch, bin_label[:, 0])
        ce_yaw =ce_loss(yaw, bin_label[:, 1])
        ce_roll = ce_loss(roll, bin_label[:, 2])
        
        mse_pitch = mse_loss(pyr[:, 0], cont_label[:, 0])
        mse_yaw = mse_loss(pyr[:, 1], cont_label[:, 1])
        mse_roll = mse_loss(pyr[:, 2], cont_label[:, 2])

        pitch_mae = mx.nd.sum(mx.nd.abs(pyr[:, 0]-cont_label[:, 0])).asscalar()
        yaw_mae = mx.nd.sum(mx.nd.abs(pyr[:, 1]-cont_label[:, 1])).asscalar()
        roll_mae = mx.nd.sum(mx.nd.abs(pyr[:, 2]-cont_label[:, 2])).asscalar()
    
    else:
        idx_tensor = mx.nd.array([idx for idx in range(66)], ctx=_ctx)
        pitch, yaw, roll = outputs[:, :66], outputs[:, 66:66*2], outputs[:, 66*2:]
        ce_pitch = ce_loss(pitch, bin_label[:, 0])
        ce_yaw =ce_loss(yaw, bin_label[:, 1])
        ce_roll = ce_loss(roll, bin_label[:, 2])
        
        pitch_pre = mx.nd.sum(mx.nd.softmax(pitch, 1)*idx_tensor, 1)*3-99
        yaw_pre = mx.nd.sum(mx.nd.softmax(yaw, 1)*idx_tensor, 1)*3-99
        roll_pre = mx.nd.sum(mx.nd.softmax(roll, 1)*idx_tensor, 1)*3-99

        mse_pitch = mse_loss(pitch_pre, cont_label[:, 0])
        mse_yaw = mse_loss(yaw_pre, cont_label[:, 1])
        mse_roll = mse_loss(roll_pre, cont_label[:, 2])
        
        pitch_mae = mx.nd.sum(mx.nd.abs(pitch_pre-cont_label[:, 0])).asscalar()
        yaw_mae = mx.nd.sum(mx.nd.abs(yaw_pre-cont_label[:, 1])).asscalar()
        roll_mae = mx.nd.sum(mx.nd.abs(roll_pre-cont_label[:, 2])).asscalar()


    loss_pitch = ce_pitch +  args.alpha*mse_pitch
    loss_yaw = ce_yaw + args.alpha*mse_yaw
    loss_roll = ce_roll + args.alpha*mse_roll

    return (loss_pitch, loss_yaw, loss_roll), (pitch_mae, yaw_mae, roll_mae)


def train(args):
    _ctx = mx.gpu(args.gpu)
    # get data
    train_loader, val_loader = get_data(args)
    # get net
    net = get_net(_ctx, args)
 
    # optimizer
    # lr_decay = 0.1
    # lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.strip().split(',')]
    # optimizer = 'sgd'
    # optimizer_params = {'learning_rate': args.lr, 'wd':args.wd, 'momentum': args.momentum}
    # trainer = mx.gluon.Trainer(net.collect_params(), optimizer=optimizer, optimizer_params=optimizer_params)
    if 'cos' in args.lr_type:
        lr_sch  = mx.lr_scheduler.CosineScheduler((args.epochs-3)*len(train_loader), args.lr, 1e-6)
        trainer = mx.gluon.Trainer(net.collect_params(), optimizer='adam', optimizer_params= {'learning_rate': args.lr, 'wd':args.wd, 'lr_scheduler': lr_sch}, )
    else:
        trainer = mx.gluon.Trainer(net.collect_params(), optimizer='adam', optimizer_params= {'learning_rate': args.lr, 'wd':args.wd}, )

    # train
    pitch_metric_loss = mx.metric.Loss()
    yaw_metric_loss = mx.metric.Loss()
    roll_metric_loss = mx.metric.Loss()
    train_history = TrainingHistory(['train-pitch', 'train-yaw', 'train-roll', 'val-pitch', 'val-yaw', 'val-roll'])
    mae_history = TrainingHistory(['train-pitch', 'train-yaw', 'train-roll', 'train-mae', 'val-pitch', 'val-yaw', 'val-roll', 'val-mae'])
    best_mae, best_epoch = np.inf, 0
    
    for epoch in range(args.epochs):
        tic = time.time()
        btic = time.time()
        pitch_metric_loss.reset()
        yaw_metric_loss.reset()
        roll_metric_loss.reset()
        total, pitch_mae, yaw_mae, roll_mae = 0, 0, 0, 0
        
        # if epoch in lr_decay_epoch:
        #     trainer.set_learning_rate(trainer.learning_rate*lr_decay)
            
 
        for i, batch in enumerate(train_loader):
            data= batch[0].as_in_context(_ctx)
            bin_label=batch[1].as_in_context(_ctx)
            cont_label=batch[2].as_in_context(_ctx)
            total += len(cont_label)
            with mx.autograd.record():
                outputs=net(data)
                loss_pyr, mae = cal_loss(outputs, bin_label, cont_label, _ctx, args)

            mx.autograd.backward([*loss_pyr])
            trainer.step(args.bs)
            pitch_metric_loss.update(0, loss_pyr[0])
            yaw_metric_loss.update(0, loss_pyr[1])
            roll_metric_loss.update(0, loss_pyr[2])
            pitch_mae += mae[0]
            yaw_mae += mae[1]
            roll_mae += mae[2]
 
            if not (i+1)%args.log_interval:
                sp = args.bs*args.log_interval/(time.time()-btic)
                train_loss = (pitch_metric_loss.get()[1],  yaw_metric_loss.get()[1], roll_metric_loss.get()[1])
                print('Epoch[%03d] Batch[%03d/%03d] Speed: %.2f samples/sec, Loss:(pitch, yaw, roll)/(%.3f, %.3f, %.3f)'%(epoch, 
                    i, len(train_loader), sp, *train_loss))
                btic = time.time()
        
        train_loss = (pitch_metric_loss.get()[1], yaw_metric_loss.get()[1], roll_metric_loss.get()[1])
        mae_ = (pitch_mae/total, yaw_mae/total, roll_mae/total)
        train_mae = (*mae_, sum([*mae_])/3)
        val_loss, val_mae = val(net, _ctx, val_loader, args)
        train_history.update([*train_loss, *val_loss])
        mae_history.update([*train_mae, *val_mae])
        print('Epoch[%03d] train: MAE:(pitch, yaw, roll, mean)/(%.3f, %.3f, %.3f, %.3f), Cost=%d sec, lr=%f'%(epoch, *train_mae, time.time()-tic, trainer.learning_rate))
        print('Epoch[%03d] val  : MAE:(pitch, yaw, roll, mean)/(%.3f, %.3f, %.3f, %.3f), Loss:(pitch, yaw, roll)/(%.3f, %.3f, %.3f)'%(epoch, *val_mae, *val_loss))
        
        # if (epoch+1)%2==0:
        #     print('save model!')
        #     net.export('%s/pose'%(save_root), epoch=epoch)
        if val_mae[3]<best_mae:
            print('Min val mean MAE! save model!')
            best_mae = val_mae[3]
            best_epoch = epoch
            net.export('%s/best_pose'%(save_root), epoch=0)

    print('\n'*2+'Min mean MAE: %.3f, Epoch: %.3f'%(best_mae, best_epoch))
    # max_ys = [max(i) for i in train_history.history.values()]
    # train_history.plot(save_path='%s/loss_log.png'%(save_root), labels=train_history.labels, y_lim=(0, max(max_ys)))
    # max_ys = [max(i) for i in mae_history.history.values()]
    # mae_history.plot(save_path='%s/mae_log.png'%(save_root), labels=mae_history.labels, y_lim=(0, max(max_ys)))
 
def val(net, _ctx, val_loader, args):
    pitch_metric_loss = mx.metric.Loss()
    yaw_metric_loss = mx.metric.Loss()
    roll_metric_loss = mx.metric.Loss()
    
    total = 0
    pitch_mae, yaw_mae, roll_mae = 0, 0, 0
    for i, batch in enumerate(val_loader):
        data= batch[0].as_in_context(_ctx)
        bin_label=batch[1].as_in_context(_ctx)
        cont_label=batch[2].as_in_context(_ctx)
        total+=len(cont_label)
        outputs = net(data)
        loss_pyr, mae = cal_loss(outputs, bin_label, cont_label, _ctx, args)
        pitch_metric_loss.update(0, loss_pyr[0])
        yaw_metric_loss.update(0, loss_pyr[1])
        roll_metric_loss.update(0, loss_pyr[2])
        pitch_mae += mae[0]
        yaw_mae += mae[1]
        roll_mae += mae[2]
    mae_ = (pitch_mae/total, yaw_mae/total, roll_mae/total)
    val_loss = (pitch_metric_loss.get()[1],  yaw_metric_loss.get()[1], roll_metric_loss.get()[1])
    return val_loss, (*mae_, sum([*mae_])/3)

if __name__ == "__main__":
    args = get_args()
    print(args)
    save_root = osp.join(args.save, args.prefix)
    if not osp.exists(save_root):
        os.makedirs(osp.join(save_root))
    train(args)


