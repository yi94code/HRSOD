#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/sigmoid_multi_label_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void SigmoidMultiLabelLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, const int label_size, const int bottom_dim, Dtype negative_scale, Dtype* label_vector_data, Dtype* loss) {
  int num = nthreads / bottom_dim;
  CUDA_KERNEL_LOOP(index, nthreads) {
    // Set label_vector_data to zeros
    int n = index / bottom_dim;
    int c = index % bottom_dim;
    label_vector_data[n * bottom_dim + c] = Dtype(0);
  }
  __syncthreads();
  CUDA_KERNEL_LOOP(index, nthreads) {
    if (index < num * label_size) {
      int sample_id = index / label_size;
      int label_id = index % label_size;
      int label_value = static_cast<int>(label[sample_id * label_size + label_id]);
      if (label_value > 0) {
	label_vector_data[sample_id * bottom_dim + label_value - 1] = Dtype(1);
      }
    }
  }
  __syncthreads();
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / bottom_dim;
    int c = index % bottom_dim;
    Dtype* cur_label_vector_data = label_vector_data + n * bottom_dim + c;
    const Dtype* cur_prob_data = prob_data + n * bottom_dim + c;
    loss[n * bottom_dim + c] = - cur_label_vector_data[0] * log(max(cur_prob_data[0], Dtype(kLOG_THRESHOLD))) - negative_scale * (1 - cur_label_vector_data[0]) * log(max(1 - cur_prob_data[0], Dtype(kLOG_THRESHOLD)));
  }
}

template <typename Dtype>
void SigmoidMultiLabelLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  Dtype* label_vector_data = label_vector_.mutable_gpu_data();
  int n = prob_.num();
  int count = prob_.count();

  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  SigmoidMultiLabelLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, prob_data, label, label_size_, bottom_dim_, negative_scale_, label_vector_data, loss_data);
  Dtype loss;
  caffe_gpu_asum(count, loss_data, &loss);
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  top[0]->mutable_cpu_data()[0] = loss / count;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  } else if (top.size() == 3) {
    top[1]->ShareData(prob_);
    top[2]->ShareData(label_vector_);
  }
}

template <typename Dtype>
__global__ void SigmoidMultiLabelLossBackwardGPU(const int nthreads, const Dtype* prob,
          const Dtype* label_vector_data, const int bottom_dim, Dtype negative_scale, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / bottom_dim;
    const int c = index % bottom_dim;
    const Dtype label_value = label_vector_data[n * bottom_dim + c];
    const Dtype prob_value = prob[n * bottom_dim + c];
    bottom_diff[n * bottom_dim + c] = -label_value * (1 - prob_value) + negative_scale * (1 - label_value) * prob_value;
  }
}

template <typename Dtype>
void SigmoidMultiLabelLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* label_vector_data = label_vector_.gpu_data();
    const int num = prob_.num(); 
    const int count = prob_.count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SigmoidMultiLabelLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, prob_data, label_vector_data, bottom_dim_,  negative_scale_, bottom_diff);

    const Dtype loss_weight = top[0]->cpu_diff()[0] / num / bottom_dim_;
    caffe_gpu_scal(count, loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidMultiLabelLossLayer);

}  // namespace caffe
