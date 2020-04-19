#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/smooth_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {
  template <typename Dtype>
  void project_simplex(const Dtype* v, const int n, const Dtype mu, const Dtype z, const Dtype dummy_max_value, Dtype* w, Dtype* dummy_w) {
    double theta, w_tmp;
    int *U, *G, *L;
    U = new int [n];
    G = new int [n];
    L = new int [n];
    Dtype* v_tmp = new Dtype[n];
    Dtype max_value = Dtype(-FLT_MAX);
    for (int i = 0; i < n; i++) {
      v_tmp[i] = v[i] / (mu + Dtype(FLT_MIN) );
      if (v_tmp[i] > max_value) {
	max_value = v_tmp[i];
      }
      U[i] = i;
    }
    double s = 0, ds = 0, ro = 0, dro = 0;
    Dtype dummy_max_tmp = dummy_max_value / (mu + Dtype(FLT_MIN));

    if (dummy_max_tmp > 0 && dummy_max_tmp > max_value) {
      s = dummy_max_tmp;
      ro = 1;
      dummy_w[0] = Dtype(1);
    } else {
      dummy_w[0] = Dtype(0);
    }

    int n_U, n_G, n_L; 
    n_U = n;
    while (n_U > Dtype(0)) {
      int k = n_U-1;
      n_G = 0; n_L =0;
      ds = 0;
      for(int i = 0; i < n_U; i++) {
        if (v_tmp[U[i]] >= v_tmp[U[k]]) {
          G[n_G++] = U[i];
          ds += double(v_tmp[U[i]]);
        } else {
          L[n_L++] = U[i];
        }
      }
      dro = double(n_G);
      if ((s+ds) -(ro + dro) * double(v_tmp[U[k]]) < z) {
        s += ds; ro += dro;
        delete [] U;
        U = L;
        n_U = n_L;
        L = new int [n_U];
      } else {
        delete [] U;
        U = G;
        n_U = n_G -1;
        G = new int [n_U];
      }
    }
    delete [] U;
    delete [] G;
    delete [] L;
    theta = (s-z) / (ro + DBL_MIN);
    for(int i = 0; i < n; i++) {
      w_tmp = double(v_tmp[i]) - theta;
      w_tmp = w_tmp > 0 ? w_tmp : double(0);
      w[i] = Dtype(w_tmp);
    } 
    if (dummy_w[0] > 0) {
      dummy_w[0] = double(dummy_max_tmp) - theta;
      dummy_w[0] = dummy_w[0] > 0 ? dummy_w[0] : Dtype(0);
    }
    //delete [] v_tmp;
  }
  template
  void project_simplex<float>(const float* v, const int n, const float mu, const float z, const float dummy_max_value, float* w, float* dummy_w);
  template 
  void project_simplex<double>(const double* v, const int n, const double mu, const double z, const double dummy_max_value, double* w, double* dummy_w);

  template <typename Dtype>
  void SmoothPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LayerParameter param = this->layer_param_;
    SmoothPoolingParameter pool_param = param.smooth_pooling_param();
    z_ = pool_param.z();
    fix_smooth_ = pool_param.fix_smooth();
    max_value_ = pool_param.max_value();
    unique_smooth_ = pool_param.unique_smooth(); 
    has_smooth_blobs_ = pool_param.has_smooth_blobs();
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    dim_ = height_ * width_;
    CHECK((bottom.size() > 1 || has_smooth_blobs_) && !(bottom.size() > 1 && has_smooth_blobs_)) << "Smooth parameters should be provided by eitehr bottom blobs or layer parameter blobs but not both.";

    if (bottom.size() > 1) {
      // bottom blob provides smooth parameters
      if (unique_smooth_) {
        CHECK(bottom[1]->width() == 1 && bottom[1]->height() == 1 && bottom[1]->channels() == 1) << "The size of smooth parameters is wrong.";
      } else {
        CHECK(bottom[1]->width() == 1 && bottom[1]->height() == 1 && bottom[1]->channels() == channels_) << "The size of smooth parameters is wrong.";
      }
      smooth_ = bottom[1];
    } else {
      this->blobs_.resize(1);
      if (unique_smooth_) {
        this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1)); 
      } else {
        this->blobs_[0].reset(new Blob<Dtype>(1, channels_, 1, 1)); 
      }
      shared_ptr<Filler<Dtype> > smooth_filler(GetFiller<Dtype>(pool_param.smooth_filler()));
      smooth_filler->Fill(this->blobs_[0].get());
      smooth_ = this->blobs_[0].get();

      this->param_propagate_down_.resize(1, true);
    }
    weight_.Reshape(num_, channels_, height_, width_);
    w_norm_.Reshape(num_, channels_, 1, 1);
  }

  template <typename Dtype>
  void SmoothPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
    << "corresponding to (num, channels, height, width)";
    CHECK_EQ(channels_, bottom[0]->channels()) << "Channel number for SmoothPooling layer should be fixed after layer setup.";
    num_ = bottom[0]->num();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    dim_ = height_ * width_;
    weight_.Reshape(num_ , channels_, height_, width_);
    w_norm_.Reshape(num_, channels_, 1, 1);
    if (!has_smooth_blobs_) {
      // smooth blobs is provided by bottom[1]
      smooth_ = bottom[1];
    }
    top[0]->Reshape(num_, channels_, 1, 1);
    if (top.size() == 2) {
    // softmax output
    top[1]->ReshapeLike(weight_);
  }
  }

  template <typename Dtype>
  void SmoothPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    Dtype* weight_data = weight_.mutable_cpu_data();

    Dtype* w_norm_data = w_norm_.mutable_cpu_data();
    const Dtype* smooth_data = smooth_->cpu_data();
    Blob<Dtype> dummy_w(1,1,1,1);
    Dtype* dummy_w_data = dummy_w.mutable_cpu_data();
    // First compute weight_ for each num and channel
    // Second weighted average for each channel
    for (int n = 0; n < num_; n++) {
      for (int c = 0; c < channels_; c++) {
        const Dtype* cur_bottom = bottom_data + bottom[0]->offset(n, c);
        Dtype* cur_top = top_data + top[0]->offset(n, c);
        Dtype* cur_weight = weight_data + weight_.offset(n, c);
	Dtype* cur_w_norm = w_norm_data + w_norm_.offset(n, c);
        Dtype cur_smooth;
        if (has_smooth_blobs_) {
          cur_smooth = unique_smooth_ ? smooth_data[0] : smooth_data[c];
        } else {
          cur_smooth = unique_smooth_ ? smooth_data[n] : smooth_data[n * channels_ + c];
        }
	cur_smooth = std::max(cur_smooth, Dtype(0.01));
        project_simplex(cur_bottom, dim_, cur_smooth, z_, max_value_, cur_weight, dummy_w_data);
	cur_w_norm[0] = caffe_cpu_dot(dim_, cur_weight, cur_weight);
	cur_w_norm[0] += dummy_w_data[0] * dummy_w_data[0];
        cur_top[0] = caffe_cpu_dot(dim_, cur_bottom, cur_weight);
	cur_top[0] += max_value_* dummy_w_data[0];
	if (!fix_smooth_) { 
	  cur_top[0] -= Dtype(0.5) * cur_smooth * cur_w_norm[0];
	}
      }
    }
    if (top.size() == 2) {
      top[1]->ShareData(weight_);
    }
  }

template <typename Dtype>
void SmoothPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* weight_data = weight_.cpu_data();
  const Dtype* w_norm_data = w_norm_.cpu_data();
  if (propagate_down[0]) {
    // Gradient with respect to bottom [0]
    // set bottom[0] diff to zeors
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    for (int n = 0; n < num_; n++) {
      for (int c = 0; c < channels_; c++) {
        const Dtype* cur_weight_data = weight_data + weight_.offset(n, c);
        Dtype* cur_bottom_diff = bottom_diff + bottom[0]->offset(n, c);
        const Dtype* cur_top_diff = top_diff + top[0]->offset(n, c); 
        caffe_axpy(dim_, *cur_top_diff, cur_weight_data,  cur_bottom_diff);
      }
    }
  }

  if (!fix_smooth_) {
    if (!has_smooth_blobs_ && propagate_down[1]) {
      // Gradient with respect to bottom[1]
      Dtype* smooth_diff = smooth_->mutable_cpu_diff();
      caffe_set(smooth_->count(), Dtype(0), smooth_diff);
      if (unique_smooth_) {
	for (int n = 0; n < num_; n++) {
	  Dtype* cur_smooth_diff = smooth_diff + n;
	  for (int c = 0; c < channels_; c++) {
	    const Dtype* cur_top_diff = top_diff + top[0]->offset(n, c); 
	    const Dtype* cur_w_norm = w_norm_data + w_norm_.offset(n, c);
	    *cur_smooth_diff += -Dtype(0.5) * cur_w_norm[0] * cur_top_diff[0];
	  }
	}
      } else {
	for (int n = 0; n < num_; n++) {
	  for (int c = 0; c < channels_; c++) {
	    Dtype* cur_smooth_diff = smooth_diff + smooth_->offset(n, c);
	    const Dtype* cur_top_diff = top_diff + top[0]->offset(n, c); 
	    const Dtype* cur_w_norm = w_norm_data + w_norm_.offset(n, c);
	    *cur_smooth_diff = -Dtype(0.5) * cur_w_norm[0] * cur_top_diff[0];
	  }
	}
      }
    } else if (has_smooth_blobs_ && this->param_propagate_down_[0]) {
      // Gradient with respect to smooth_ param
      Dtype* smooth_diff = smooth_->mutable_cpu_diff();
      //caffe_set(smooth_->count(), Dtype(0), smooth_diff);
      if (unique_smooth_) {
	for (int n = 0; n < num_; n++) {
	  Dtype* cur_smooth_diff = smooth_diff;
	  for (int c = 0; c < channels_; c++) {
	    const Dtype* cur_top_diff = top_diff + top[0]->offset(n, c); 
	    const Dtype* cur_w_norm = w_norm_data + w_norm_.offset(n, c);
	    *cur_smooth_diff += -Dtype(0.5) * cur_w_norm[0] * cur_top_diff[0];

	  }
	}
      } else {
	for (int n = 0; n < num_; n++) {
	  for (int c = 0; c < channels_; c++) {
	    Dtype* cur_smooth_diff = smooth_diff + c;
	    const Dtype* cur_top_diff = top_diff + top[0]->offset(n, c); 
	    const Dtype* cur_w_norm = w_norm_data + w_norm_.offset(n, c);
	    *cur_smooth_diff += -Dtype(0.5) * cur_w_norm[0] * cur_top_diff[0];
	  }
	}
      }
    }
  }

}

template <typename Dtype> 
void SmoothPoolingLayer<Dtype>::UpdateSmooth(const Dtype smooth) {
  if (!has_smooth_blobs_) {
    return;
  }
  LOG(INFO) << "Current smooth: " << smooth;
  switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_set(smooth_->count(), smooth, smooth_->mutable_cpu_data());
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      caffe_gpu_set(smooth_->count(), smooth, smooth_->mutable_gpu_data());
#else
      NO_GPU;
#endif
      break;
  }
}


#ifdef CPU_ONLY
STUB_GPU(SmoothPoolingLayer);
#endif

INSTANTIATE_CLASS(SmoothPoolingLayer);

}  // namespace caffe
