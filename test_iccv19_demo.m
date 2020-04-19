close all; clear; clc;
addpath('utils/');
init_iccv19_demo; %set parameters in this file
rng(0);

%% Main loop
for ii=1:length(imnames)
    fprintf('Processing Img:%d/%d\n', ii, length(imnames));
    %% read image
    im = imread(sprintf('%s%s', imgRoot, imnames(ii).name));
    [height, width, ch] = size(im);
    if ch ~= 3
        im = repmat(im, [1,1,3]);
    end
    globalinput{1} = prepare_img(im, false);
    %% forward pass
    global_sal_map = network_forward_sal(net, globalinput);
    global_sal_map = global_sal_map(:,:,2);
    %%stage2
    if opt ==0
        global_sal_map = imresize(global_sal_map, [height, width]);
    end
    if opt == 1
        im = imresize(im, [1024, 1024]);
        global_sal_map = imresize(global_sal_map, [1024, 1024]);
        height_ori = height;
        width_ori = width;
        [height, width, ~] = size(im);
    end

    tmp = zeros(height, width);
    flag = zeros(height, width);

if mode == 1
     sal= uint8(im2double(global_sal_map));
     sal255=uint8(global_sal_map*255);

     [h, w, ~] = size(sal);  
     [row,col] = find(sal255>50 & sal255<200);
   if ~isempty (col)            
     W = max(col)-min(col);
     D = 384;
     N = ceil(W/D)+3;
     d = round(W/N);
     index = cell(1,N+1);
     for x_id=1:N+1
         x = min((min(col)+d*(x_id-1)),max(col));
         relax = unidrnd(140)-70;
         crop_size = D+relax;
         index{x_id} = find(col==x);
         num=length(find(col==x));
         if num~=0
         center_x = x;
         d_heit = max(row(index{x_id}))-min(row(index{x_id}));
         if d_heit<0.5*crop_size
             center_y = zeros(1,1);
             j = unidrnd(num);
             center_y(1) = row(index{x_id}(j));
         elseif (d_heit>0.5*crop_size)&&(d_heit<crop_size)
             center_y = zeros(1,2);
             center_y(1) = min(row(index{x_id}));
             center_y(2) = max(row(index{x_id}));
         elseif d_heit>crop_size
             N_h = ceil(d_heit/crop_size);
                 d_h = round(d_heit/N_h);
                 center_y = zeros(1,N_h+1);
                 for y_id=1:N_h+1
                 center_y(y_id) = min((min(row)+d_h*(y_id-1)),max(row));
                 end
         end
         x_bgin = max(1, center_x - round(crop_size*0.5));
         x_end = min(w, x_bgin+crop_size);
         y_bgin = zeros(1,length(center_y));
         y_end = zeros(1,length(center_y));
         for p = 1:length(center_y)
             y_bgin(p) = max(1, center_y(p) - round(crop_size*0.5));    
             y_end(p) = min(h, y_bgin(p)+crop_size);
             patch = cell(1,5);
             patch_im = cell(1,5);
             patch{p}  = imcrop(sal,[x_bgin y_bgin(p) crop_size crop_size]);
             patch_im{p}  = imcrop(im,[x_bgin y_bgin(p) crop_size crop_size]);
             [p_h, p_w, p_ch] = size(patch_im{p});
                if p_ch ~= 3
                    patch_im{p} = repmat(patch_im{p}, [1,1,3]);
                end
                size_im1=48;
                size_im2=96;
                input{1} = prepare_img384(patch_im{p}, false);
                input{2} = prepare_image(patch{p},size_im1);
                input{3} = prepare_image(patch{p},size_im2);

                %% forward pass
                [sal_map1] = network_forward_sal(net2, input);
                sal_map1 = sal_map1(:,:,2);
                sal_map1 = imresize(sal_map1, [p_h, p_w]);
                %% merge
                x = x_bgin;
                y = y_bgin(p);
                x2 = min(width, x+crop_size);
                y2 = min(height, y+crop_size);
                for ri=y:y2
                    for rj=x:x2
                        tmp(ri,rj)= tmp(ri,rj)+sal_map1(1+ri-y,1+rj-x);
                        flag(ri,rj) = flag(ri,rj)+1;
                    end
                end
         end
         end            
     end


     tmp(flag~=0)=tmp(flag~=0)./flag(flag~=0);
     global_sal_map(flag~=0)=tmp(flag~=0);
     
     stage2_sal_map = global_sal_map;
   clear index;
   else
       stage2_sal_map = global_sal_map;
   end
end

if mode == 0
        sal= uint8(im2double(global_sal_map));
                size_im1=48;
                size_im2=96;
                input{1} = prepare_img384(im, false);
                input{2} = prepare_image(sal,size_im1);
                input{3} = prepare_image(sal,size_im2);

                %% forward pass
                [sal_map1] = network_forward_sal(net2, input);
                sal_map1 = sal_map1(:,:,2);
                sal_map1 = imresize(sal_map1, [height, width]);
end
%%stage3
if mode == 1

    size_sal=1024;
     stage2_sal=uint8(stage2_sal_map*255);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%resize
       input2{1} = prepare_img1024(im, false);
        input2{2} = prepare_image(stage2_sal,size_sal);

    %% forward pass
    [stage3_sal_map] = network_forward_sal(net3, input2);
    stage3_sal_map = stage3_sal_map(:,:,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%resize
    if opt==0
    stage3_sal_map = imresize(stage3_sal_map, [height, width]);
    end
    if opt==1
    stage3_sal_map = imresize(stage3_sal_map, [height_ori, width_ori]);
    end
    imwrite(stage3_sal_map, [res_path imnames(ii).name(1:end-3) 'png']);

end
if mode == 0

    size_sal=384;
     stage2_sal=uint8(sal_map1*255);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%resize
      input2{1} = prepare_img384(im, false);
      input2{2} = prepare_image(stage2_sal,size_sal);

    %% forward pass
    [stage3_sal_map] = network_forward_sal(net3, input2);
    stage3_sal_map = stage3_sal_map(:,:,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%resize
    stage3_sal_map = imresize(stage3_sal_map, [height, width]);
    imwrite(stage3_sal_map, [res_path imnames(ii).name(1:end-3) 'png']);

end

end
caffe.reset_all;
    
