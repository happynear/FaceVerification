caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

ROIx = 19:82;
ROIy = 19:82;
% ROIx = 1:100;
% ROIy = 1:100;

% ROIx = 1:64;
% ROIy = 1:64;
height = length(ROIx);
width = length(ROIy);

allPairs = [same_pair;diff_pair];
meanC = caffe.read_mean('D:\ThirdPartyLibrary\caffe\examples\siamese\web_mean.proto');
net = caffe.Net('D:\ThirdPartyLibrary\caffe\examples\siamese\CASIA_deploy.prototxt','D:\ThirdPartyLibrary\caffe\examples\siamese\siamese_iter_912000.caffemodel', 'test');
% meanC = caffe.read_mean('C:\Users\feng\Documents\Tencent Files\438797086\FileRecv\meanfile');
% net = caffe.Net('C:\Users\feng\Documents\Tencent Files\438797086\FileRecv\deep_ID1.prototxt','C:\Users\feng\Documents\Tencent Files\438797086\FileRecv\DP__iter_120000.caffemodel','test');
num = size(allPairs,1);
AllFeature1 = zeros(320,num);
AllFeature2 = zeros(320,num);
for i = 1 : floor(num/100)
    disp([i floor(num/100)]);
    J = zeros(height,width,1,100,'single');
    for j = 1 : 100
        I = imread(allPairs{(i-1)*100+j,1});
        I = single(I') - meanC;
        I = I(ROIx,ROIy);
        J(:,:,1,j) = I/128;
%         J(:,:,1,j) = I(end:-1:1,:);
    end;
    H={J};
    f = net.forward(H);
    f = f{1};
    AllFeature1(:,(i-1)*100+1:i*100) = reshape(f,[size(AllFeature1,1),100]);
end;
J = zeros(height,width,1,100,'single');
for j = 1 : num - floor(num/100) * 100
    I = imread(allPairs{floor(num/100) * 100+j,1});
    I = single(I') - meanC;
    I = I(ROIx,ROIy);
    J(:,:,1,j) = I/128;
end;
H={J};
f = net.forward(H);
f=f{1};
f = squeeze(f);
AllFeature1(:,floor(num/100) * 100+1:num) = f(:,1 : num - floor(num/100) * 100);

for i = 1 : floor(num/100)
    disp([i floor(num/100)]);
    J = zeros(height,width,1,100,'single');
    for j = 1 : 100
        I = imread(allPairs{(i-1)*100+j,2});
        I = single(I') - meanC;
        I = I(ROIx,ROIy);
        J(:,:,1,j) = I/128;
    end;
    H={J};
    f = net.forward(H);
    f = f{1};
    AllFeature2(:,(i-1)*100+1:i*100) = reshape(f,[size(AllFeature2,1),100]);
end;
J = zeros(height,width,1,100,'single');
for j = 1 : num - floor(num/100) * 100
    I = imread(allPairs{floor(num/100) * 100+j,2});
    I = single(I') - meanC;
    I = I(ROIx,ROIy);
    J(:,:,1,j) = I/128;
end;
H={J};
f = net.forward(H);
f=f{1};
f = squeeze(f);
AllFeature2(:,floor(num/100) * 100+1:num) = f(:,1 : num - floor(num/100) * 100);