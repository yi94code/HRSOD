#include <algorithm>
#include <vector>

#include "caffe/layers/output_decay_layer.hpp"

namespace caffe {

template <typename Dtype>
void OutputDecayLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK(bottom.size() == 1 && top.size() == 1) << "OutputDecayLayer takes exactly one input and output blob";
  CHECK(bottom[0] == top[0]) << "OutputDecayLayer conduct in-place computation.";
}

template <typename Dtype>
void OutputDecayLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  temp_.ReshapeLike(*bottom[0]);
}
 

template <typename Dtype>
void OutputDecayLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    const Dtype decay = this->layer_param_.output_decay_param().output_decay();
    if (this->layer_param_.output_decay_param().regularization_type() == "L2") {
      caffe_axpy(count, decay, bottom_data, bottom_diff);

    } else if (this->layer_param_.output_decay_param().regularization_type() == "L1") {
      caffe_cpu_sign(count,
            bottom_data,
            temp_.mutable_cpu_data());
      caffe_axpy(count,
            decay,
            temp_.cpu_data(),
            bottom_diff);
    } else {
      LOG(FATAL) << "Unknown Regularization type " << this->layer_param_.output_decay_param().regularization_type();
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(OutputDecayLayer);
#endif

INSTANTIATE_CLASS(OutputDecayLayer);
REGISTER_LAYER_CLASS(OutputDecay);
}  // namespace caffe
