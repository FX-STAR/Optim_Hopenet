
# mxnet2caffe
## caffe_plugin_layer
- Relu6 layer (op:clip in (0, 6)) 
- Broadcastmul (op:broadcast_mul)
  ```Shell
  # you can get detail about add relu6 in caffe here:
  https://blog.csdn.net/JR_Chan/article/details/94584068
  All layer files in caffe_plugin_include and caffe_plugin_src
  ```
## convert
```shell
cd mxnet2caffe
mkdir model (put mxnet model in this file)
```
### 1. json2prototxt
```shell
python mxnet2caffe.py --save model/ --prefix best_pose --prototxt pose.prototxt --caffemodel pose.caffemodel --trans net
```
### 2. params2caffemodel
```shell
python mxnet2caffe.py --save model/ --prefix best_pose --prototxt pose.prototxt --caffemodel pose.caffemodel --trans weight
```
