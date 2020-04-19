#include <cfloat>
#include <vector>
#include <stdio.h>
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxForward(const int nthreads, const Dtype* bottom_data_a,
    const Dtype* bottom_data_b, const int blob_idx, Dtype* top_data,
    int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    if (bottom_data_a[index] > bottom_data_b[index]) {
      // only update for very first bottom_data blob (blob_idx == 0)
      if (blob_idx == 0) {
        maxval = bottom_data_a[index];
        top_data[index] = maxval;
        maxidx = blob_idx;
        mask[index] = maxidx;
      }
    } else {
      maxval = bottom_data_b[index];
      top_data[index] = maxval;
      maxidx = blob_idx + 1;
      mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void ProdForward(const int nthreads, const Dtype* bottom_data_a, const Dtype* bottom_data_b, const int dim, const int channels, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
   int num = index / (dim * channels); 
   int c = (index / dim) % channels;
   int id = index % dim;
   top_data[(num * channels + c) * dim + id] = bottom_data_a[(num * channels + c) * dim + id] * bottom_data_b[num * dim + id];
  }
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int* mask = NULL;
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    if (cross_channel_) {
      const int dim = top[0]->count(2);
      const int channels = top[0]->channels();
      ProdForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), dim, channels, top_data);
      for (int i = 2; i < bottom.size(); ++i) {
	ProdForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_data, bottom[i]->gpu_data(), dim, channels, top_data);
      }
    } else {
      caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
	  top_data);
      for (int i = 2; i < bottom.size(); ++i) {
	caffe_gpu_mul(count, top_data, bottom[i]->gpu_data(), top_data);
      }
    }
    //for (int n = 0; n < bottom[0]->num(); n++) {
    //  for (int c = 0; c < bottom[0]->channels(); c++) {
    //    const Dtype* cur_b0 = bottom[0]->cpu_data() + bottom[0]->offset(n, c);
    //    const Dtype* cur_b1 = bottom[1]->cpu_data() + bottom[1]->offset(n);
    //    const Dtype* cur_b2 = bottom[2]->cpu_data() + bottom[2]->offset(n);
    //    const Dtype* cur_top = top[0]->cpu_data() + top[0]->offset(n, c);
    //    printf("n = %d, c = %d\n", n, c);
    //    for (int i = 0; i < bottom[0]->count(2); i++) {
    //      printf("bottom0: %f, bottom1: %f, bottom2: %f\n top: %f \n", float(cur_b0[i]), float(cur_b1[i]), float(cur_b2[i]), float(cur_top[i]));
    //    }
    //  }
    //}

    break;
  case EltwiseParameter_EltwiseOp_SUM:
    caffe_gpu_set(count, Dtype(0.), top_data);
    // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom.size(); ++i) {
      caffe_gpu_axpy(count, coeffs_[i], bottom[i]->gpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    mask = max_idx_.mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 0, top_data, mask);
    for (int i = 2; i < bottom.size(); ++i) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      MaxForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_data, bottom[i]->gpu_data(), i-1, top_data, mask);
    }
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Dtype>
__global__ void MaxBackward(const int nthreads, const Dtype* top_diff,
    const int blob_idx, const int* mask, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype gradient = 0;
    if (mask[index] == blob_idx) {
      gradient += top_diff[index];
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void ProdBackwardStable(const int nthreads, const Dtype* top_diff, const Dtype* temp_diff, const Dtype* bottom_a_data, const int channels, const int dim, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
   int num = index / dim;
   int id = index % dim;
   const Dtype* cur_top_diff = top_diff + num * channels * dim +id;
   const Dtype* cur_temp_diff = temp_diff + num * dim + id;
   const Dtype* cur_bottom_a_data = bottom_a_data + num * channels * dim + id;
   Dtype* cur_bottom_diff = bottom_diff + num * dim + id;
   Dtype diff = 0;
   for (int i = 0; i < channels; i++) {
     diff += cur_top_diff[i * dim] * cur_bottom_a_data[i * dim];
   }
   cur_bottom_diff[0] = diff * cur_temp_diff[0];
  }
}

template <typename Dtype>
__global__ void ProdBackward(const int nthreads, const Dtype* top_data, const Dtype* top_diff, const Dtype* bottom_data, const int channels, const int dim, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
   int num = index / dim;
   int id = index % dim;
   const Dtype* cur_top_diff = top_diff + num * channels * dim +id;
   const Dtype* cur_top_data = top_data + num * channels * dim +id;
   const Dtype* cur_bottom_data = bottom_data + num * dim + id;
   Dtype* cur_bottom_diff = bottom_diff + num * dim + id;
   Dtype diff = 0;
   for (int i = 0; i < channels; i++) {
     diff += cur_top_diff[i * dim] * cur_top_data[i * dim] / cur_bottom_data[0]; 
   }
   cur_bottom_diff[0] = diff;
  }
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int* mask = NULL;
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
        if (cross_channel_) {
	  const int dim = top[0]->count(2);
	  const int channels = top[0]->channels();
	  const int reference_count = bottom[1]->count();
	  Blob<Dtype> temp_blob(top[0]->num(), 1, top[0]->height(), top[0]->width());
	  Dtype* temp_blob_diff = temp_blob.mutable_gpu_diff();
	  caffe_gpu_set(reference_count, Dtype(1), temp_blob_diff);
	  if (stable_prod_grad_) {
	    bool initialized = false;
	    for (int j = 1; j < bottom.size(); ++j) {
	      if (i == j) { continue; }
	      if (!initialized) {
		caffe_copy(reference_count, bottom[j]->gpu_data(), temp_blob_diff);
		initialized = true;
	      } else {
		caffe_gpu_mul(reference_count, temp_blob_diff, bottom[j]->gpu_data(), temp_blob_diff);
	      }
	    }
	    if (i ==0) {
	      ProdForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, temp_blob_diff, dim, channels, bottom_diff);
	    } else {
	      const Dtype* bottom_a_data = bottom[0]->gpu_data();
	      ProdBackwardStable<Dtype><<<CAFFE_GET_BLOCKS(reference_count), CAFFE_CUDA_NUM_THREADS>>>(reference_count, top_diff, temp_blob_diff, bottom_a_data, channels, dim, bottom_diff);
	    }
	  } else {
	    if (i == 0) {
	      caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
	      caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
	    } else {
	      const Dtype* bottom_a_data = bottom[0]->gpu_data();
	      ProdBackward<Dtype><<<CAFFE_GET_BLOCKS(reference_count), CAFFE_CUDA_NUM_THREADS>>>(reference_count, top_data, top_diff, bottom_data, channels, dim, bottom_diff);
	    }
	   
	  }
	} else {
	  if (stable_prod_grad_) {
	    bool initialized = false;
	    for (int j = 0; j < bottom.size(); ++j) {
	      if (i == j) { continue; }
	      if (!initialized) {
		caffe_copy(count, bottom[j]->gpu_data(), bottom_diff);
		initialized = true;
	      } else {
		caffe_gpu_mul(count, bottom[j]->gpu_data(), bottom_diff,
		    bottom_diff);
	      }
	    }
	  } else {
	    caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
	  }
	  caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
	}
        break;
      case EltwiseParameter_EltwiseOp_SUM:
        if (coeffs_[i] == Dtype(1.)) {
          caffe_copy(count, top_diff, bottom_diff);
        } else {
          caffe_gpu_scale(count, coeffs_[i], top_diff, bottom_diff);
        }
        break;
      case EltwiseParameter_EltwiseOp_MAX:
        mask = max_idx_.gpu_data();
        MaxBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, i, mask, bottom_diff);
        break;
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EltwiseLayer);

}  // namespace caffe
