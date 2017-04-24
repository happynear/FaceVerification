% caffe.reset_all();
if ~exist('imageList','var')
    BLUFR_config = 'D:\face project\BLUFR\config\lfw\blufr_lfw_config.mat';
    load(BLUFR_config);
end;
lfw_root = 'C:\datasets\lfw-aligned\';
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

% ROIx = 19:82;
% ROIy = 19:82;
ROIx = 1:96;
ROIy = 1:112;
feature_dim = 512;
mean_value = 128;
scale = 0.0078125;
batch_size = 100;
total_length = length(imageList);
total_iter = ceil(total_length / batch_size);

% ROIx = 1:64;
% ROIy = 1:64;
height = length(ROIx);
width = length(ROIy);

net = caffe.Net('D:\face project\experiment\96_112_l2_distance\face_deploy.prototxt','D:\face project\experiment\96_112_l2_distance\face_train_test_iter_15000.caffemodel', 'test');%face_model_my
Descriptors = zeros(total_iter * batch_size, feature_dim);
for i = 1 : total_iter
    disp([i total_iter]);
    J = zeros(height,width,3,batch_size,'single');
    for j = 1 : batch_size
        if (i-1)*batch_size+j > total_length
            break;
        end;
        file_name = imageList{(i-1)*batch_size+j};
        find_zero = strfind(file_name,'0');
        folder = file_name(1:find_zero(1)-2);
        
        I = imread(fullfile(lfw_root, folder, file_name));
        I = permute(I,[2 1 3]);
        I = I(:,:,[3 2 1]);
        I = I(ROIx,ROIy,:);
        I = single(I) - mean_value;
        J(:,:,:,j) = I*scale;
%         J(:,:,1,j) = I(end:-1:1,:);
    end;
    f1 = net.forward({J});
    f1 = f1{1};
    Descriptors((i-1)*batch_size+1:i*batch_size,:) = reshape(f1,[feature_dim,batch_size])';
end;
Descriptors = Descriptors(1:total_length,:);
save('lfw_feature.mat','Descriptors');