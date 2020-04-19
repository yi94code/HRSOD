#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/smooth_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "iostream"
#include "stdio.h"

namespace caffe {


  template <typename Dtype>
    __global__ void SmoothPoolForward(const int nthreads, const Dtype* bottom_data, const int num, const int channels, const int dim, int* index_data, Dtype* value_data, const bool unique_smooth, const bool has_smooth_blobs, const Dtype z, const Dtype dummy_max_value, const Dtype* smooth_data, Dtype* weight, Dtype* w_norm_data, const bool fix_smooth, Dtype* top_data) {
      CUDA_KERNEL_LOOP(index, nthreads) {
	const int n = index / channels;
	const int c = index % channels;
	int *U, *G, *L, *ind_tmp;
	U = index_data + 3 * dim * index;
	G = U + dim;
	L = G + dim;
	const Dtype* cur_bottom = bottom_data + dim * index;
	Dtype* v_tmp = value_data + dim * index;
	Dtype* w = weight + dim * index;
	Dtype* w_norm = w_norm_data + index;
	Dtype* o = top_data + index;
	double theta,  w_tmp;
	Dtype mu;
	if (has_smooth_blobs) {
	  if (unique_smooth) {
	    mu = max(smooth_data[0], Dtype(0.01));
	  } else {
	    mu = max(smooth_data[c], Dtype(0.01));
	  }
	} else {
	  if (unique_smooth) {
	    mu = max(smooth_data[n], Dtype(0.01));
	  } else {
	    mu = max(smooth_data[index], Dtype(0.01));
	  }
	}
	
	Dtype max_value = Dtype(-FLT_MAX);
	Dtype dummy_w;
	for (int i = 0; i < dim; i++) {
	  v_tmp[i] = cur_bottom[i] / (mu + Dtype(FLT_MIN)) ;
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
	  dummy_w = Dtype(1);
	} else {
	  dummy_w = Dtype(0);
	}
	int n_U, n_G, n_L; 
	n_U = dim;
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
	    ind_tmp = U;
	    U = L;
	    n_U = n_L;
	    L = ind_tmp;
	  } else {
	    ind_tmp = U;
	    U = G;
	    n_U = n_G -1;
	    G = ind_tmp;
	  }
	}
	theta = (s-double(z)) / (ro + DBL_MIN);

        if (dummy_w > Dtype(0)) {
	  dummy_w = double(dummy_max_tmp) - theta;
	  dummy_w = dummy_w > 0 ? dummy_w : Dtype(0);
	  o[0] = dummy_w * dummy_max_value;
	  w_norm[0] = dummy_w * dummy_w;
	} else {
	  o[0] = Dtype(0);
	  w_norm[0] = 0;
	}
	for (int i = 0; i < dim; i++) {
	  w_tmp = double(v_tmp[i]) - theta;
	  w_tmp = w_tmp > 0 ? w_tmp : double(0);
	  w[i] = Dtype(w_tmp);
	  w_norm[0] += w[i] * w[i];
	  o[0] += cur_bottom[i] * Dtype(w[i]);
	}
	if (! fix_smooth) {
	  w_norm[0] *= -Dtype(0.5);
	  o[0] += mu * w_norm[0];
	}
      }
    }



  template <typename Dtype>
    void SmoothPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
      const Dtype* bottom_data = bottom[0]->gpu_data();
      Dtype* top_data = top[0]->mutable_gpu_data();
      int count = top[0]->count();
      const Dtype* smooth_data = smooth_->gpu_data();
      Dtype* weight_data =  weight_.mutable_gpu_data();
      Dtype* w_norm_data = w_norm_.mutable_gpu_data();
      Blob<int> index_set(3*num_, channels_, height_, width_);
      int* index_data = index_set.mutable_gpu_data();
      Blob<Dtype> value(num_, channels_, height_, width_);
      Dtype* value_data = value.mutable_gpu_data();
      SmoothPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, num_, channels_, dim_, 
	  index_data, value_data, unique_smooth_, has_smooth_blobs_, z_, max_value_, smooth_data, weight_data, w_norm_data, fix_smooth_, top_data);
      if (top.size() == 2) {
	top[1]->ShareData(weight_);
      }
      CUDA_POST_KERNEL_CHECK;
    }
    
    template <typename Dtype> 
    __global__ void SmoothPoolBackwardBottom(const int nthreads, const Dtype* top_diff, const Dtype* weight, const int channels, const int dim,  Dtype* bottom_diff) {
      CUDA_KERNEL_LOOP(index, nthreads) {
	const int n = index / channels / dim;
	const int c = (index / dim) % channels;
	const int id = index % (dim);
	const Dtype* cur_top_diff = top_diff + n * channels + c;
	const Dtype* cur_weight = weight + (n * channels + c) * dim + id;
	Dtype* cur_bottom_diff = bottom_diff + (n * channels + c) * dim + id;
	cur_bottom_diff[0] = cur_top_diff[0] * cur_weight[0];

      }
    }


     template <typename Dtype> 
    __global__ void SmoothPoolBackwardUnique(const int nthreads, const Dtype* top_diff, const Dtype* w_norm, const int channels, Dtype* smooth_diff) {
      CUDA_KERNEL_LOOP(index, nthreads) {
	const Dtype* cur_top_diff = top_diff + index * channels;
	const Dtype* cur_w_norm = w_norm + index * channels;
	Dtype* cur_smooth_diff = smooth_diff + index;
	cur_smooth_diff[0] = 0;
	for (int i = 0; i < channels; i++) {
	  cur_smooth_diff[0] += cur_w_norm[i] * cur_top_diff[i];
	}
	//cur_smooth_diff[0] *= -Dtype(0.5);
      }
    }


  template <typename Dtype>
    __global__ void SmoothPoolBackward(const int nthreads, const Dtype* top_diff, const int num, const int channels, const Dtype* w_norm_data, Dtype* smooth_diff) {
      CUDA_KERNEL_LOOP(index, nthreads) {
	const Dtype* cur_top_diff = top_diff + index;
	const Dtype* cur_w_norm = w_norm_data + index;

	for (int i = 0; i < num; i++) {
	  smooth_diff[index] += cur_top_diff[i*channels] * cur_w_norm[i*channels];
	}
	//smooth_diff[index] *= -Dtype(0.5); 
      }
    }

  template <typename Dtype>
    __global__ void caffe_gpu_hadamard_product(const int nthreads, const Dtype* a, const Dtype* b, Dtype* c) {
      CUDA_KERNEL_LOOP(index, nthreads) {
	c[index] = a[index] * b[index];
	}
    }



  template <typename Dtype>
    void SmoothPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* weight_data = weight_.gpu_data();
      const Dtype* w_norm_data = w_norm_.gpu_data();
      if (propagate_down[0]) {
	//Gradient with respect to bottom [0]
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	int count = bottom[0]->count();
	SmoothPoolBackwardBottom<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, weight_data, channels_, dim_,  bottom_diff);
      }

      if (!fix_smooth_) {
	if (!has_smooth_blobs_ && propagate_down[1]) {
	  Dtype* smooth_diff = smooth_->mutable_gpu_diff();
	  if (unique_smooth_) {
	    SmoothPoolBackwardUnique<Dtype><<<CAFFE_GET_BLOCKS(num_), CAFFE_CUDA_NUM_THREADS>>>(num_, top_diff, w_norm_data, channels_, smooth_diff); 
	  } else {
	    int count = top[0]->count();
	    caffe_gpu_hadamard_product<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, w_norm_data, smooth_diff);
	  }
	} else if (has_smooth_blobs_ && this->param_propagate_down_[0]) {
	  // Gradient with respect to smooth_ param
	  //caffe_gpu_set(smooth_->count(), Dtype(0), smooth_diff);
	  if (unique_smooth_) {
	    int count = top[0]->count();
	    Dtype* smooth_cpu_diff = smooth_->mutable_cpu_diff();
	    caffe_gpu_dot(count, w_norm_data, top_diff, smooth_cpu_diff);
	    //smooth_cpu_diff[0] *= -Dtype(0.5);
	  } else {
	    Dtype* smooth_diff = smooth_->mutable_gpu_diff();
	    SmoothPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(channels_), CAFFE_CUDA_NUM_THREADS>>>(channels_, top_diff, num_, channels_,  w_norm_data, smooth_diff);
	  }
	}
      }
      CUDA_POST_KERNEL_CHECK;
    }
  INSTANTIATE_LAYER_GPU_FUNCS(SmoothPoolingLayer);
}  // namespace caffe
