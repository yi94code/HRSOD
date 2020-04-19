function out = prepare_img512(im, mean_file)
if size(im, 3) ~= 3
    im = repmat(im, [1,1,3]);
end
im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
if mean_file
    im_mean = load('ilsvrc_2012_mean.mat');
%     IMAGE_DIM = 256;
        IMAGE_DIM = 1024;

    im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM]);  % resize im_data
%im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');
    out = im_data - im_mean.mean_data;  % subtract mean_data (already in W x H x C, BGR)
else
    out = bsxfun(@minus, im_data, reshape([103.939, 116.779, 123.68],[1,1,3]));
end

%     out = imresize(out, [256 256], 'bilinear');  % resize im_data
        out = imresize(out, [1024 1024], 'bilinear');  % resize im_data

end
