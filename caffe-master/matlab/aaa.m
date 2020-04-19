clear
caffe.set_mode_cpu();
net_model = ['../examples/smooth_pool/deploy2.prototxt'];
net = caffe.Net(net_model, 'train');
input = 300*rand(40, 40, 512, 1);
out_diff = rand(1, 1, 512, 1);


tic;
out = net.forward({single(input)});
% toc
% tic;
net.backward({single(out_diff)});
toc

caffe.set_mode_gpu();

tic;
out = net.forward({single(input)});
% toc
% tic;
net.backward({single(out_diff)});
toc
caffe.reset_all