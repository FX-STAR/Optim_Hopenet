#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/broadcastmul_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
/*
template <typename Dtype>
void BroadcastmulLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){
  
  NeuronLayer<Dtype>::LayerSetUp(bottom,top);
  power_ = this->layer_param_.my_neuron_param().power();
  b_ = this->layer_param_.my_neuron_param().b();
 }
 */

// Compute y = x^power + b
template <typename Dtype>
void BroadcastmulLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  //caffe_powx(count, bottom[0]->cpu_data(), Dtype(power_), top_data);
  caffe_copy(count, bottom_data, top_data);

  /*for (int i = 0; i < count; ++i)
   {
    top_data[i] = bottom_data[i];
   }
*/

}
 
template <typename Dtype>
void BroadcastmulLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom){

  const int count = bottom[0]->count();
  const Dtype* top_diff = top[0]->cpu_diff();
  if(propagate_down[0]){
   
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(count, top_diff, bottom_diff);
  }
 
}
 

#ifdef CPU_ONLY
STUB_GPU(BroadcastmulLayer);
#endif


INSTANTIATE_CLASS(BroadcastmulLayer);
REGISTER_LAYER_CLASS(Broadcastmul);
 
}// namespace caffe

