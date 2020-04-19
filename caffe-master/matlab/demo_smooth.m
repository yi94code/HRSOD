% clear
caffe.set_mode_gpu();
net_file = ['../examples/smooth_pool/demo_train.prototxt'];
solver_file = ['../examples/smooth_pool/demo_solver.prototxt'];
% net = caffe.Net(net_model, 'train');
% input = rand(10, 10, 2, 1);
% input(1:20) = 30 * rand(20,1);
input(1,1,1,1) = 100;%max(input(:));
input(1,1,2,1) = 100;%max(input(:));
solver = caffe.Solver(solver_file);




tic;
for i = 1:800
out = solver.net.forward({single(input)});
% toc
% tic;
% tar = zeros(1,1,2,1);
% tar(1) = 1;
% out_diff = out{1} - tar;
out_diff = zeros(1,1,2,1);
out_diff(1) = -1/out{1}(1);
solver.net.empty_net_param_diff();
a = solver.net.backward({single(out_diff)});

u_diff =  solver.net.params('pool1',1).get_diff();
u = solver.net.params('pool1',1).get_data();
p = solver.net.blobs('pool1').get_data();
p_diff = solver.net.blobs('pool1').get_diff();
fprintf('out = %f \t u = %f \t diff = %08f\n', out{1}(1), u, u_diff);
w = a{1}(:,:,1)/(p_diff(1)+eps);
w(:,:,2) = a{1}(:,:,2)/(p_diff(2)+eps);
solver.apply_update();
end
toc
% 
% caffe.set_mode_gpu();
% 
% tic;
% out = net.forward({single(input)});
% % toc
% % tic;
% net.backward({single(out_diff)});
% toc
caffe.reset_all