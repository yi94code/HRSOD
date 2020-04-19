#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/overlap_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void OverlapAccuracyForwardGPU(const int nthreads, const int dim, const Dtype* prediction, const Dtype* label, Dtype* pre, Dtype* recall) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const Dtype* cur_prediction = prediction + index * dim;
    const Dtype* cur_label = label + index * dim;
    Dtype count_pre, count_gt, count_it;
    count_pre = 0; count_gt = 0; count_it = 0;
    for (int i = 0; i < dim; i++) {
      if(cur_label[i] > 0.5 && cur_prediction[i] >= 0.5) {
	count_pre++;
	count_gt++;
	count_it++;
      } else if (label[i] >0.5) {
	count_gt++;
      } else if (cur_prediction[i] >= 0.5) {
	count_pre++;
      }
    }
    pre[index] = count_it / (count_pre + Dtype(FLT_MIN));
    recall[index] = count_it / (count_gt + Dtype(FLT_MIN));
  }
}

template <typename Dtype>
void OverlapAccuracyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int dim = bottom[0]->count(1);
  Dtype* pre_data = pre_.mutable_gpu_data();
  Dtype* recall_data = recall_.mutable_gpu_data();
  const Dtype* prediction = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data(); 
  OverlapAccuracyForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(num),
      CAFFE_CUDA_NUM_THREADS>>>(num, dim, prediction, label, pre_data, recall_data);
  Dtype P, R;
  caffe_gpu_asum(num, pre_data, &P);
  caffe_gpu_asum(num, recall_data, &R);
  top[0]->mutable_cpu_data()[0] = P / num;
  if (top.size() >= 2) {
    top[1]->mutable_cpu_data()[0] = R / num;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(OverlapAccuracyLayer);

}  // namespace caffe
