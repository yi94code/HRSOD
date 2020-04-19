% function [ out_map] = network_forward_sal(net, im)
function [ out_map1] = network_forward_sal(net, input)

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%     [height, width, ~] = size(im);
%     input = prepare_img(im, false);
    out_map = net.forward(input);
%     fea1=net.blobs('convres').get_data();
%     fea2=net.blobs('mask-convres').get_data();
%      fea3=net.blobs('res4').get_data();
%     fea4=net.blobs('mask-res4').get_data();
%        fea5=net.blobs('pre0_full-sm-tiled').get_data();

%    out_map1 = out_map{1};
%          out_map2 = out_map{20};
    out_map1 = out_map{1};

%    fea1 = permute(fea1, [2,1,3]);
%    fea2 = permute(fea2, [2,1,3]);
%    fea3 = permute(fea3, [2,1,3]);
%    fea4 = permute(fea4, [2,1,3]);
%    fea5 = permute(fea5, [2,1,3]);

    out_map1 = permute(out_map1, [2,1,3]);
%          out_map2 = permute(out_map2, [2,1,3]);
         
%    fea1 = imresize(fea1, [height, width]);
%    fea2 = imresize(fea2, [height, width]);
%    fea3 = imresize(fea3, [height, width]);
%    fea4 = imresize(fea4, [height, width]);
%    fea5 = imresize(fea5, [height, width]);


%     out_map1 = imresize(out_map1, [height, width]);
%          out_map2 = imresize(out_map2, [height, width]);

% 
%     a1 = net.blobs('upscore1').get_data;
%     a1 = permute(a1, [2,1,3]);
%     a2 = net.blobs('upscore2').get_data;
%     a2 = permute(a2, [2,1,3]);
%     a3 = net.blobs('upscore3').get_data;
%     a3 = permute(a3, [2,1,3]);
%     a4 = net.blobs('upscore4').get_data;
%     a4 = permute(a4, [2,1,3]);
%     figure(10);
%     subplot(2,2,1);
%     imagesc(a1(:,:,2));
%     subplot(2,2,2);
%     imagesc(a2(:,:,2));
%     subplot(2,2,3);
%     imagesc(a3(:,:,2));
%     subplot(2,2,4);
%     imagesc(a4(:,:,2));
end