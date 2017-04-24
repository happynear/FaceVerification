caffe.reset_all();
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

% ROIx = 1:64;
% ROIy = 1:64;
height = length(ROIx);
width = length(ROIy);

allPairs = [same_pair;diff_pair];
% meanC = caffe.read_mean('D:\ThirdPartyLibrary\caffe\examples\siamese\mean.proto');
net = caffe.Net('D:\face project\experiment\96_112_l2_distance\face_deploy.prototxt','D:\face project\experiment\96_112_l2_distance\face_model.caffemodel', 'test');%
% net = caffe.Net('caffe_proto\center_loss_ms.prototxt','caffe_proto\center_loss_ms.caffemodel', 'test');
% net = caffe.Net('D:\face project\experiment\Model_facecenter_Align_vertical_96_112\face_deploy.prototxt','E:\downloads\face_model.caffemodel', 'test');
num = size(allPairs,1);
AllFeature1 = zeros(feature_dim,num);
AllFeature2 = zeros(feature_dim,num);
for i = 1 : floor(num/100)
    disp([i floor(num/100)]);
    J = zeros(height,width,3,100,'single');
    for j = 1 : 100
        I = imread(allPairs{(i-1)*100+j,1});
        I = permute(I,[2 1 3]);
        I = I(:,:,[3 2 1]);
        I = I(ROIx,ROIy,:);
        I = single(I) - mean_value;
        J(:,:,:,j) = I*scale;
%         J(:,:,1,j) = I(end:-1:1,:);
    end;
    f1 = net.forward({J});
    f1 = f1{1};
    AllFeature1(1:feature_dim,(i-1)*100+1:i*100) = reshape(f1,[size(AllFeature1,1),100]);
%     layer_conv52 = net.blob_vec(net.name2blob_index('pool5'));
%     conv52 = layer_conv52.get_data();
%     sum(conv52(:)>0) /320/100
end;
J = zeros(height,width,3,100,'single');
for j = 1 : num - floor(num/100) * 100
    I = imread(allPairs{floor(num/100) * 100+j,1});
    I = permute(I,[2 1 3]);
    I = I(:,:,[3 2 1]);
    I = I(ROIx,ROIy,:);
    I = single(I) - mean_value;
    J(:,:,:,j) = I*scale;
end;
f1 = net.forward({J});
f1=f1{1};
f1 = squeeze(f1);
AllFeature1(1:feature_dim,floor(num/100) * 100+1:num) = f1(:,1 : num - floor(num/100) * 100);

for i = 1 : floor(num/100)
    disp([i floor(num/100)]);
    J = zeros(height,width,3,100,'single');
    for j = 1 : 100
        I = imread(allPairs{(i-1)*100+j,2});
        I = permute(I,[2 1 3]);
        I = I(:,:,[3 2 1]);
        I = I(ROIx,ROIy,:);
        I = single(I) - mean_value;
        J(:,:,:,j) = I*scale;
    end;
    f1 = net.forward({J});
    f1 = f1{1};
    AllFeature2(1:feature_dim,(i-1)*100+1:i*100) = reshape(f1,[size(AllFeature2,1),100]);
end;
J = zeros(height,width,3,100,'single');
for j = 1 : num - floor(num/100) * 100
    I = imread(allPairs{floor(num/100) * 100+j,2});
    I = permute(I,[2 1 3]);
    I = I(:,:,[3 2 1]);
    I = I(ROIx,ROIy,:);
    I = single(I) - mean_value;
    J(:,:,:,j) = I*scale;
end;
f1 = net.forward({J});
f1=f1{1};
f1 = squeeze(f1);
AllFeature2(1:feature_dim,floor(num/100) * 100+1:num) = f1(:,1 : num - floor(num/100) * 100);
% caffe.reset_all();