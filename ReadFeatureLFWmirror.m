caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

% ROIx = 19:82;
% ROIy = 19:82;
ROIx = 1:96;
ROIy = 1:112;

% ROIx = 1:64;
% ROIy = 1:64;
height = length(ROIx);
width = length(ROIy);

allPairs = [same_pair;diff_pair];
% meanC = caffe.read_mean('D:\ThirdPartyLibrary\caffe\examples\siamese\mean.proto');
net = caffe.Net('D:\face project\experiment\Model_facecenter_Align_vertical_96_112_resnet_max\face_deploy.prototxt','D:\face project\experiment\Model_facecenter_Align_vertical_96_112_resnet_max\face_train_test_iter_30000.caffemodel', 'test');
% net = caffe.Net('D:\face project\experiment\Model_facecenter_Align_vertical_96_112\face_deploy.prototxt','E:\downloads\face_model.caffemodel', 'test');
num = size(allPairs,1);
AllFeature1 = zeros(1024,num);
AllFeature2 = zeros(1024,num);
for i = 1 : floor(num/100)
    disp([i floor(num/100)]);
    J = zeros(height,width,3,100,'single');
    for j = 1 : 100
        I = imread(allPairs{(i-1)*100+j,1});
        I = permute(I,[2 1 3]);
        I = I(:,:,[3 2 1]);
        I = I(ROIx,ROIy,:);
        I = single(I) - 127.5;
        J(:,:,:,j) = I/128;
%         J(:,:,1,j) = I(end:-1:1,:);
    end;
    f1 = net.forward({J});
    f1 = f1{1};
%     J1 = J(:,:,:,1);
%     figure(4);
%     imshow(uint8(permute(J1(:,:,[3 2 1]),[2 1 3]) *128 + 127.5))
    for j=1:100
        J(:,:,1,j) = flipud(J(:,:,1,j));
        J(:,:,2,j) = flipud(J(:,:,2,j));
        J(:,:,3,j) = flipud(J(:,:,3,j));
    end;
%     J1 = J(:,:,:,1);
%     figure(5);
%     imshow(uint8(permute(J1(:,:,[3 2 1]),[2 1 3]) *128 + 127.5))
    f2 = net.forward({J});
    f2 = f2{1};
    AllFeature1(1:512,(i-1)*100+1:i*100) = reshape(f1,[size(AllFeature1,1) / 2,100]);
    AllFeature1(513:1024,(i-1)*100+1:i*100) = reshape(f2,[size(AllFeature1,1) / 2,100]);
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
    I = single(I) - 127.5;
    J(:,:,:,j) = I/128;
end;
f1 = net.forward({J});
f1=f1{1};
f1 = squeeze(f1);
for j=1: num - floor(num/100) * 100
    J(:,:,1,j) = flipud(J(:,:,1,j));
    J(:,:,2,j) = flipud(J(:,:,2,j));
    J(:,:,3,j) = flipud(J(:,:,3,j));
end;
f2 = net.forward({J});
f2 = f2{1};
f2 = squeeze(f2);
AllFeature1(1:512,floor(num/100) * 100+1:num) = f1(:,1 : num - floor(num/100) * 100);
AllFeature1(513:1024,floor(num/100) * 100+1:num) = f2(:,1 : num - floor(num/100) * 100);

for i = 1 : floor(num/100)
    disp([i floor(num/100)]);
    J = zeros(height,width,3,100,'single');
    for j = 1 : 100
        I = imread(allPairs{(i-1)*100+j,2});
        I = permute(I,[2 1 3]);
        I = I(:,:,[3 2 1]);
        I = I(ROIx,ROIy,:);
        I = single(I) - 127.5;
        J(:,:,:,j) = I/128;
    end;
    f1 = net.forward({J});
    f1 = f1{1};
    for j=1:100
        J(:,:,1,j) = flipud(J(:,:,1,j));
        J(:,:,2,j) = flipud(J(:,:,2,j));
        J(:,:,3,j) = flipud(J(:,:,3,j));
    end;
    f2 = net.forward({J});
    f2 = f2{1};
    AllFeature2(1:512,(i-1)*100+1:i*100) = reshape(f1,[size(AllFeature2,1)/2,100]);
    AllFeature2(513:1024,(i-1)*100+1:i*100) = reshape(f2,[size(AllFeature2,1)/2,100]);
end;
J = zeros(height,width,3,100,'single');
for j = 1 : num - floor(num/100) * 100
    I = imread(allPairs{floor(num/100) * 100+j,2});
    I = permute(I,[2 1 3]);
    I = I(:,:,[3 2 1]);
    I = I(ROIx,ROIy,:);
    I = single(I) - 127.5;
    J(:,:,:,j) = I/128;
end;
f1 = net.forward({J});
f1=f1{1};
f1 = squeeze(f1);
for j=1: num - floor(num/100) * 100
    J(:,:,1,j) = flipud(J(:,:,1,j));
    J(:,:,2,j) = flipud(J(:,:,2,j));
    J(:,:,3,j) = flipud(J(:,:,3,j));
end;
f2 = net.forward({J});
f2 = f2{1};
f2 = squeeze(f2);
AllFeature2(1:512,floor(num/100) * 100+1:num) = f1(:,1 : num - floor(num/100) * 100);
AllFeature2(513:1024,floor(num/100) * 100+1:num) = f2(:,1 : num - floor(num/100) * 100);