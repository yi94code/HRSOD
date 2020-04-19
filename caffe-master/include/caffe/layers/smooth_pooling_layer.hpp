#ifndef CAFFE_SMOOTH_POOLING_LAYER_HPP_
#define CAFFE_SMOOTH_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <float.h>

namespace caffe {

/**
 * @brief Pools the input image by taking the max, average, etc. within regions.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SmoothPoolingLayer : public Layer<Dtype> {
 public:
  explicit SmoothPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SmoothPooling"; }
  virtual inline int MinTopBlobs() const { return 1; }
  
  virtual inline int MaxTopBlobs() const { return 2;}
  virtual void UpdateSmooth(const Dtype smooth);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool has_smooth_blobs_, unique_smooth_, fix_smooth_;
  int num_;
  int channels_;
  int height_, width_;
  int dim_;
  Dtype z_;
  Dtype max_value_;
  Blob<Dtype> weight_;
  Blob<Dtype> w_norm_;
  Blob<Dtype>*  smooth_;
};
template <typename Dtype>
  void project_simplex(const Dtype* v, int n, Dtype mu, Dtype z, Dtype* w);
}  // namespace caffe

#endif  // CAFFE_SMOOT_POOLING_LAYER_HPP_
