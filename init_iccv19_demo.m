%% Add path and init caffe
caffe_root = '/home/zengyi/Downloads/HRSOD-master/caffe-master/';%add your caffe path here
addpath([caffe_root 'matlab/'],genpath('./external'), 'util/');
caffe.reset_all;
gpu_id = 0;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);

mode = 1; %Set it to be 1 for high-resolution datatsets or 0 for low-resolution datasets
version = 1;  %Set it to be 1 for Ours-DH or 0 for Ours-D
opt = 1; %Set it to be 1 for super high-resolution datasets(e.g., HRSOD) or 0 for normal high-res dataset(e.g., DAVIS-S)

if version == 0
model_weights = './models/Training_on_DUTS/GSN.caffemodel';
model_weights2 = './models/Training_on_DUTS/LRN.caffemodel';
model_weights3='./models/Training_on_DUTS/GLFN.caffemodel';
if mode == 1
model_def3 = './prototxt/deploy_GLFN_DUTS_high_res.prototxt';
end
if mode == 0
model_def3 = './prototxt/deploy_GLFN_DUTS_low_res.prototxt';
end
end

if version == 1
model_weights = './models/Training_on_DUTSandHRSOD-TR/GSN.caffemodel';
model_weights2 = './models/Training_on_DUTSandHRSOD-TR/LRN.caffemodel';
model_weights3='./models/Training_on_DUTSandHRSOD-TR/GLFN.caffemodel';
if mode == 1
 model_def3 = './prototxt/deploy_GLFN_high_res.prototxt';
end
if mode == 0
 model_def3 = './prototxt/deploy_GLFN_low_res.prototxt';
end
end

model_def = './prototxt/deploy-GSN.prototxt';
model_def2 = './prototxt/deploy-LRN.prototxt';
phase = 'test';
net = caffe.Net(model_def, model_weights, phase);
net2 = caffe.Net(model_def2, model_weights2, phase);
net3 = caffe.Net(model_def3, model_weights3, phase);

%% input output path

% imgRoot='/home/zengyi/Desktop/DAVIS1080/im1080_select/';
% imgRoot='/media/home/zengyi/Downloads/datasets/HKU-IS/HKU-IS-Image/';
% imgRoot='/home/zengyi/Downloads/AFNet-master/test-Image/';
imgRoot='/media/home/zengyi/Downloads/datasets/HRSOD_release/HRSOD_test/'; %set your image path here

res_path = './results/';
if ~isdir(res_path)
    mkdir(res_path);
end
imnames=dir([imgRoot '*' '.jpg']);







