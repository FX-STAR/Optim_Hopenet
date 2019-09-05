### 
# @Description: In User Settings Edit
 # @Author: your name
 # @Date: 2019-08-31 22:56:05
 # @LastEditTime: 2019-09-05 09:49:04
 # @LastEditors: Please set LastEditors
 ###
nohup python -u train.py --bs 128 --lr 0.001 --alpha 1 --lr_type cos --version small --width_mult 1 --use_fc 1 --net v3 --gpu 0 --prefix test > weight/test.log  2>&1 &
