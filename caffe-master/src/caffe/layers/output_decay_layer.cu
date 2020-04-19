#include <algorithm>
#include <vector>

#include "caffe/layers/output_decay_layer.hpp"
namespace caffe {
template <typename Dtype>
void OutputDecayLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    const Dtype decay = this->layer_param_.output_decay_param().output_decay();
    if (this->layer_param_.output_decay_param().regularization_type() == "L2") {
      caffe_gpu_axpy(count, decay, bottom_data, bottom_diff);
    } else if (this->layer_param_.output_decay_param().regularization_type() == "L1") {
      caffe_gpu_sign(count,
            bottom_data,
            temp_.mutable_gpu_data());
      caffe_gpu_axpy(count,
            decay,
            temp_.gpu_data(),
            bottom_diff);
    } else {
      LOG(FATAL) << "Unknown Regularization type " << this->layer_param_.output_decay_param().regularization_type();
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(OutputDecayLayer);

}  // namespace caffe
