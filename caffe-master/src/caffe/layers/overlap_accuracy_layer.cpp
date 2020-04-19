#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/overlap_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void OverlapAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  pre_.Reshape(bottom[0]->num(), 1, 1, 1);
  recall_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void OverlapAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  if (top.size() >= 2) {
    top[1]->Reshape(top_shape);
  }
  pre_.Reshape(bottom[0]->num(), 1, 1, 1);
  recall_.Reshape(bottom[0]->num(), 1, 1, 1);
}
//
//template <typename Dtype>
//void AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//    const vector<Blob<Dtype>*>& top) {
//  Dtype accuracy = 0;
//  const Dtype* bottom_data = bottom[0]->cpu_data();
//  const Dtype* bottom_label = bottom[1]->cpu_data();
//  const int dim = bottom[0]->count() / outer_num_;
//  const int num_labels = bottom[0]->shape(label_axis_);
//  vector<Dtype> maxval(top_k_+1);
//  vector<int> max_id(top_k_+1);
//  if (top.size() > 1) {
//    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
//    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
//  }
//  int count = 0;
//  for (int i = 0; i < outer_num_; ++i) {
//    for (int j = 0; j < inner_num_; ++j) {
//      const int label_value =
//          static_cast<int>(bottom_label[i * inner_num_ + j]);
//      if (has_ignore_label_ && label_value == ignore_label_) {
//        continue;
//      }
//      if (top.size() > 1) ++nums_buffer_.mutable_cpu_data()[label_value];
//      DCHECK_GE(label_value, 0);
//      DCHECK_LT(label_value, num_labels);
//      // Top-k accuracy
//      std::vector<std::pair<Dtype, int> > bottom_data_vector;
//      for (int k = 0; k < num_labels; ++k) {
//        bottom_data_vector.push_back(std::make_pair(
//            bottom_data[i * dim + k * inner_num_ + j], k));
//      }
//      std::partial_sort(
//          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
//          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
//      // check if true label is in top k predictions
//      for (int k = 0; k < top_k_; k++) {
//        if (bottom_data_vector[k].second == label_value) {
//          ++accuracy;
//          if (top.size() > 1) ++top[1]->mutable_cpu_data()[label_value];
//          break;
//        }
//      }
//      ++count;
//    }
//  }
//
//  // LOG(INFO) << "Accuracy: " << accuracy;
//  top[0]->mutable_cpu_data()[0] = accuracy / count;
//  if (top.size() > 1) {
//    for (int i = 0; i < top[1]->count(); ++i) {
//      top[1]->mutable_cpu_data()[i] =
//          nums_buffer_.cpu_data()[i] == 0 ? 0
//          : top[1]->cpu_data()[i] / nums_buffer_.cpu_data()[i];
//    }
//  }
//  // Accuracy layer should not be used as a loss function.
//}

//#ifdef CPU_ONLY
//STUB_GPU(OverlapAccuracyLayer);
//#endif

INSTANTIATE_CLASS(OverlapAccuracyLayer);
REGISTER_LAYER_CLASS(OverlapAccuracy);

}  // namespace caffe
