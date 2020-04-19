#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/sigmoid_multi_label_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class SigmoidMultiLabelLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SigmoidMultiLabelLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 15, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 8, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(1);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    int n = blob_bottom_label_->num();
    int label_size = blob_bottom_label_->channels();
    int c = blob_bottom_data_->channels();
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < label_size-3; ++j) {
	blob_bottom_label_->mutable_cpu_data()[i * label_size + j] = caffe_rng_rand() % c + 1;
      }
      for (int j = label_size-3; j < label_size; ++j) {
	blob_bottom_label_->mutable_cpu_data()[i * label_size + j] = Dtype(0);
      }
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~SigmoidMultiLabelLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SigmoidMultiLabelLayerTest, TestDtypesAndDevices);

TYPED_TEST(SigmoidMultiLabelLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_negative_scale(0.1);
  layer_param.add_loss_weight(3);
  SigmoidMultiLabelLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
