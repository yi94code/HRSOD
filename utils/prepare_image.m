function images = prepare_image(guide,size_im)
IMAGE_DIM = size_im;

% resize to fixed input size
% im = single(im);
guide=single(guide);

% mean BGR pixel
% mean_pix = [103.939, 116.779, 123.68];

% im = imresize(im, [IMAGE_DIM IMAGE_DIM]);
guide=imresize(guide,[IMAGE_DIM IMAGE_DIM]);
% RGB -> BGR
% im = im(:, :, [3 2 1]);

% oversample (4 corners, center, and their x-axis flips)
% images = zeros(IMAGE_DIM, IMAGE_DIM, 4, 1, 'single');
% images(:,:,1:3,1)=permute(im,[2 1 3]);
images=permute(guide,[2,1]);

    % mean BGR pixel subtraction
% for c = 1:3
%     images(:, :, c, :) = images(:, :, c, :) - mean_pix(c);
% end

end
