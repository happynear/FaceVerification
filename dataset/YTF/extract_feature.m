% if exist('image_list.mat','file')
%     load('image_list.mat');
% end;
if ~exist('image_list','var')
    folder = 'D:\datasets\aligned_images_DB2';
    addpath('..');
    image_list = get_image_list_in_folder(folder); 
end;
caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

feature_dim = 512;
batch_size = 100;
mean_value = 128;
scale = 0.0078125;
ROIx = 1:96;
ROIy = 1:112;
height = length(ROIx);
width = length(ROIy);

net = caffe.Net('E:\Feng\face\l2_distance_contrastive\face_deploy.prototxt','E:\Feng\face\finetune\joint loss\ip+l2\center\weight0.01\face_train_test_iter_10000.caffemodel', 'test');%face_model_my

total_image = length(image_list);
total_iter = ceil(total_image / batch_size);
features = zeros(feature_dim,total_iter*batch_size);

image_p = 1;
for i=1:total_iter
    fprintf('%d/%d\n',i, total_iter);
    J = zeros(height,width,3,batch_size,'single');
    for j = 1 : batch_size
        if image_p <= total_image
            I = imread(image_list{image_p});
            I = permute(I,[2 1 3]);
            I = I(:,:,[3 2 1]);
            I = I(ROIx,ROIy,:);
            I = single(I) - mean_value;
            J(:,:,:,j) = I*scale;
            image_p = image_p + 1;
        end;
    end;
    f1 = net.forward({J});
    f1 = f1{1};
    features(:,(i-1)*batch_size+1:i*batch_size) = f1;
end;