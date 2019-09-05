#ifndef CAFFE_BROADCASTMUL_LAYER_HPP_
#define CAFFE_BROADCASTMUL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Computes the softmax function.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class BroadcastmulLayer : public NeuronLayer<Dtype>  {
 public:
  explicit BroadcastmulLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  
   virtual inline const char* type() const { return "Broadcastmul"; }
   virtual inline int ExactNumBottomBlobs() const { return 2; }
   virtual inline int MinBottomBlobs() const { return 1; }
   virtual inline int MaxBottomBlobs() const { return 2; }
   virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_LAYER_HPP_
