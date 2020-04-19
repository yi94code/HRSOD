#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/sigmoid_multi_label_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidMultiLabelLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter sigmoid_param(this->layer_param_);
  sigmoid_param.clear_loss_weight();
  sigmoid_param.set_type("Sigmoid");
  sigmoid_layer_ = LayerRegistry<Dtype>::CreateLayer(sigmoid_param);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(&prob_);
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  CHECK(!(this->layer_param_.loss_param().has_ignore_label() || 
  this->layer_param_.loss_param().has_normalization() || 
  this->layer_param_.loss_param().has_normalize())) 
  << "MultiLabelCrossEntropyLossLayer currently does not implement ignore_label or normalization.";
  label_size_ = bottom[1]->count(1, 4);
  bottom_dim_ = bottom[0]->channels();
  label_vector_.Reshape(bottom[0]->num(), bottom_dim_, 1, 1);
  negative_scale_ = Dtype(this->layer_param_.loss_param().negative_scale());
}

template <typename Dtype>
void SigmoidMultiLabelLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  CHECK_EQ(bottom_dim_, bottom[0]->channels()) << "bottom dimmension should not be changed";
  label_size_ = bottom[1]->count(1, 4);
  label_vector_.Reshape(bottom[0]->num(), bottom_dim_, 1, 1);
  if (top.size() == 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  } else if (top.size() == 3) {
    top[1]->ReshapeLike(*bottom[0]);
    top[2]->ReshapeLike(label_vector_);
  }
}


template <typename Dtype>
void SigmoidMultiLabelLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype* label_vector_data = label_vector_.mutable_cpu_data();
  caffe_set(label_vector_.count(), Dtype(0), label_vector_data);
  int num = bottom[0]->num();
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    const Dtype* cur_label = label + i * label_size_;
    const Dtype* cur_prob = prob_data + i * bottom_dim_;
    Dtype* cur_label_vector_data = label_vector_data + i * bottom_dim_ ;
    for(int l = 0; l < label_size_; l++) {
      const int label_value = static_cast<int>(cur_label[l]);
      if (label_value == 0) {
	break;
      }
      cur_label_vector_data[label_value-1] = Dtype(1);
    }
    for (int o = 0; o < bottom_dim_; o++) {
      loss -= cur_label_vector_data[o] * log(std::max(cur_prob[o], Dtype(kLOG_THRESHOLD))) + negative_scale_ * (1 - cur_label_vector_data[o]) * log(std::max(1 - cur_prob[o], Dtype(kLOG_THRESHOLD)));
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / num / bottom_dim_;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  } else if (top.size() == 3) {
    top[1]->ShareData(prob_);
    top[2]->ShareData(label_vector_);
  }
}

template <typename Dtype>
void SigmoidMultiLabelLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label_vector_data = label_vector_.cpu_data();
    const Dtype* prob_data = prob_.cpu_data();
    int num = bottom[0]->num();
    for (int i = 0; i < num; ++i) {
    const Dtype* cur_label_vector_data = label_vector_data + i * bottom_dim_ ;
    const Dtype* cur_prob = prob_data + i * bottom_dim_;
    Dtype* cur_bottom_diff = bottom_diff + i * bottom_dim_;
      for (int j = 0; j < bottom_dim_; ++j) {
	cur_bottom_diff[j] = - cur_label_vector_data[j] * (1 - cur_prob[j]) + negative_scale_ * (1-cur_label_vector_data[j]) * cur_prob[j];
//	cur_bottom_diff[j] = 0;
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / num / bottom_dim_;
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidMultiLabelLossLayer);
#endif
INSTANTIATE_CLASS(SigmoidMultiLabelLossLayer);

}  // namespace caffe
