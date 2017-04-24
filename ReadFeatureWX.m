caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 1;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

% ROIx = 19:82;
% ROIy = 19:82;
ROIx = 9:136;
ROIy = 9:136;
feature_dim = 256;
mean_value = 0;
scale = 1 / 255;
batch_size = 100;

% ROIx = 1:64;
% ROIy = 1:64;
height = length(ROIx);
width = length(ROIy);

allPairs = [same_pair;diff_pair];
% meanC = caffe.read_mean('D:\ThirdPartyLibrary\caffe\examples\siamese\mean.proto');
% net = caffe.Net('D:\face project\face_verification_experiment\proto\LightenedCNN_B_deploy.prototxt','D:\face project\face_verification_experiment\model\LightenedCNN_B.caffemodel', 'test');%face_model_my
net = caffe.Net('D:\face project\face_verification_experiment\proto\LightenedCNN_B_deploy.prototxt','D:\face project\experiment\fine-tune\wuxiang\face_train_test_iter_14000.caffemodel', 'test');%face_model_my
% net = caffe.Net('caffe_proto\center_loss_ms.prototxt','caffe_proto\center_loss_ms.caffemodel', 'test');
% net = caffe.Net('D:\face project\experiment\Model_facecenter_Align_vertical_96_112\face_deploy.prototxt','E:\downloads\face_model.caffemodel', 'test');
num = size(allPairs,1);
AllFeature1 = zeros(feature_dim,num);
AllFeature2 = zeros(feature_dim,num);
for i = 1 : floor(num/batch_size)
    disp([i floor(num/batch_size)]);
    J = zeros(height,width,1,batch_size,'single');
    for j = 1 : batch_size
        I = imread(allPairs{(i-1)*batch_size+j,1});
        I = rgb2gray(I)';
%         I = permute(I,[2 1 3]);
%         I = I(:,:,[3 2 1]);
        I = I(ROIx,ROIy,:);
        I = single(I) - mean_value;
        J(:,:,:,j) = I*scale;
%         J(:,:,1,j) = I(end:-1:1,:);
    end;
    f1 = net.forward({J});
    f1 = f1{1};
    AllFeature1(1:feature_dim,(i-1)*batch_size+1:i*batch_size) = reshape(f1,[size(AllFeature1,1),batch_size]);
%     layer_conv52 = net.blob_vec(net.name2blob_index('pool5'));
%     conv52 = layer_conv52.get_data();
%     sum(conv52(:)>0) /320/batch_size
end;
J = zeros(height,width,1,batch_size,'single');
for j = 1 : num - floor(num/batch_size) * batch_size
    I = imread(allPairs{floor(num/batch_size) * batch_size+j,1});
    I = rgb2gray(I)';
%     I = permute(I,[2 1 3]);
%     I = I(:,:,[3 2 1]);
    I = I(ROIx,ROIy,:);
    I = single(I) - mean_value;
    J(:,:,:,j) = I*scale;
end;
f1 = net.forward({J});
f1=f1{1};
f1 = squeeze(f1);
AllFeature1(1:feature_dim,floor(num/batch_size) * batch_size+1:num) = f1(:,1 : num - floor(num/batch_size) * batch_size);

for i = 1 : floor(num/batch_size)
    disp([i floor(num/batch_size)]);
    J = zeros(height,width,1,batch_size,'single');
    for j = 1 : batch_size
        I = imread(allPairs{(i-1)*batch_size+j,2});
        I = rgb2gray(I)';
%         I = permute(I,[2 1 3]);
%         I = I(:,:,[3 2 1]);
        I = I(ROIx,ROIy,:);
        I = single(I) - mean_value;
        J(:,:,:,j) = I*scale;
    end;
    f1 = net.forward({J});
    f1 = f1{1};
    AllFeature2(1:feature_dim,(i-1)*batch_size+1:i*batch_size) = reshape(f1,[size(AllFeature2,1),batch_size]);
end;
J = zeros(height,width,1,batch_size,'single');
for j = 1 : num - floor(num/batch_size) * batch_size
    I = imread(allPairs{floor(num/batch_size) * batch_size+j,2});
    I = rgb2gray(I)';
%     I = permute(I,[2 1 3]);
%     I = I(:,:,[3 2 1]);
    I = I(ROIx,ROIy,:);
    I = single(I) - mean_value;
    J(:,:,:,j) = I*scale;
end;
f1 = net.forward({J});
f1=f1{1};
f1 = squeeze(f1);
AllFeature2(1:feature_dim,floor(num/batch_size) * batch_size+1:num) = f1(:,1 : num - floor(num/batch_size) * batch_size);
% caffe.reset_all();