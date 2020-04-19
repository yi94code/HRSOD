#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/smooth_pooling_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.

template <typename TypeParam>
class SmoothPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SmoothPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 2, 3, 3)),
        blob_bottom_2_(new Blob<Dtype>(2, 1, 1, 1)),
	blob_bottom_3_(new Blob<Dtype>(2, 2, 1, 1)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(10.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    ConstantFiller<Dtype> filler2(filler_param);
    filler2.Fill(this->blob_bottom_2_);
    filler2.Fill(this->blob_bottom_3_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~SmoothPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_bottom_3_;
    delete blob_top_;
  }


  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_bottom_3_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SmoothPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(SmoothPoolingLayerTest, TestGradientUniqueHasSmooth) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SmoothPoolingParameter* pool_param = layer_param.mutable_smooth_pooling_param();
  pool_param->set_has_smooth_blobs(true);
  pool_param->set_unique_smooth(true);
  pool_param->mutable_smooth_filler()->set_value(10);
  SmoothPoolingLayer<Dtype> layer(layer_param);
  layer.LayerSetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SmoothPoolingLayerTest, TestGradientHasSmooth) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SmoothPoolingParameter* pool_param = layer_param.mutable_smooth_pooling_param();
  pool_param->set_has_smooth_blobs(true);
  pool_param->set_unique_smooth(false);
  pool_param->mutable_smooth_filler()->set_value(10);
  SmoothPoolingLayer<Dtype> layer(layer_param);
  layer.LayerSetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SmoothPoolingLayerTest, TestGradientHasSmoothDummyMaxValue) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SmoothPoolingParameter* pool_param = layer_param.mutable_smooth_pooling_param();
  pool_param->set_has_smooth_blobs(true);
  pool_param->set_unique_smooth(false);
  pool_param->mutable_smooth_filler()->set_value(10);
  pool_param->set_z(5);
  pool_param->set_max_value(10);
  SmoothPoolingLayer<Dtype> layer(layer_param);
  layer.LayerSetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SmoothPoolingLayerTest, TestGradientUnique) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SmoothPoolingParameter* pool_param = layer_param.mutable_smooth_pooling_param();
  pool_param->set_has_smooth_blobs(false);
  pool_param->set_unique_smooth(true);
  SmoothPoolingLayer<Dtype> layer(layer_param);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  layer.LayerSetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SmoothPoolingLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SmoothPoolingParameter* pool_param = layer_param.mutable_smooth_pooling_param();
  pool_param->set_has_smooth_blobs(false);
  pool_param->set_unique_smooth(false);
  SmoothPoolingLayer<Dtype> layer(layer_param);
  this->blob_bottom_vec_.push_back(this->blob_bottom_3_);
  layer.LayerSetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
