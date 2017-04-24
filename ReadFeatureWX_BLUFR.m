% caffe.reset_all();
if ~exist('imageList','var')
    BLUFR_config = 'D:\face project\BLUFR\config\lfw\blufr_lfw_config.mat';
    load(BLUFR_config);
end;
lfw_root = 'E:\datasets\lfw-aligned-wuxiang';
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);


% ROIx = 19:82;
% ROIy = 19:82;
ROIx = 9:136;
ROIy = 9:136;
feature_dim = 256;
mean_value = 0;
scale = 1 / 255;
batch_size = 50;
total_length = length(imageList);
total_iter = ceil(total_length / batch_size);

% ROIx = 1:64;
% ROIy = 1:64;
height = length(ROIx);
width = length(ROIy);

% net = caffe.Net('D:\face project\face_verification_experiment\proto\LightenedCNN_B_deploy.prototxt','D:\face project\experiment\fine-tune\wuxiang\face_train_test_iter_14000.caffemodel', 'test');
net = caffe.Net('D:\face project\face_verification_experiment\proto\LightenedCNN_B_deploy.prototxt','D:\face project\experiment\fine-tune\wuxiang\LightenedCNN_B.caffemodel', 'test');
Descriptors = zeros(total_iter * batch_size, feature_dim);
for i = 1 : total_iter
    disp([i total_iter]);
    J = zeros(height,width,1,batch_size,'single');
    for j = 1 : batch_size
        if (i-1)*batch_size+j > total_length
            break;
        end;
        file_name = imageList{(i-1)*batch_size+j};
        find_zero = strfind(file_name,'0');
        folder = file_name(1:find_zero(1)-2);
        
        I = imread(fullfile(lfw_root, folder, file_name));
        I = rgb2gray(I)';
        I = I(ROIx,ROIy,:);
        I = single(I) - mean_value;
        J(:,:,:,j) = I*scale;
    end;
    f1 = net.forward({J});
    f1 = f1{1};
    Descriptors((i-1)*batch_size+1:i*batch_size,:) = reshape(f1,[feature_dim,batch_size])';
end;
Descriptors = Descriptors(1:total_length,:);
save('lfw_feature.mat','Descriptors');