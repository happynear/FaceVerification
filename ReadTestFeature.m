fid = fopen('G:\face-CASIA-WebFace\list.txt');
C2 = textscan(fid, '%s %d');
fclose(fid);
meanC = caffe('read_mean','D:\ThirdPartyLibrary\caffe\examples\siamese\mean.proto');

maxId = 300;

idx = find(C2{2}<=maxId);
idx = randperm(length(idx));
idx = idx(1:6000);
C{2} = C2{2}(idx);
C{1} = C2{1}(idx);

matcaffe_init(1,'D:\ThirdPartyLibrary\caffe\examples\siamese\softmax\mnist_siamese_deploy.prototxt','D:\ThirdPartyLibrary\caffe\examples\siamese\softmax\siamese_iter_58000.caffemodel');
num = length(C{2});
label = C{2};
% num = floor(num /80) * 2;
AllFeature = zeros(320,num);
for i = 1 : floor(num/100)
    disp([i floor(num/100)]);
    J = zeros(100,100,1,100,'single');
    for j = 1 : 100
        I = imread(C{1}{(i-1)*100+j});
%         I = I(end:-1:1,:);
        I = I';
%         I = I(end:-1:1,:);
        I = single(I) - meanC;
        J(:,:,1,j) = I / 128;
%         J(:,:,1,j) = I;
    end;
    H={J};
    f = caffe('forward',H);
    f = f{1};
    AllFeature(:,(i-1)*100+1:i*100) = reshape(f,[size(AllFeature,1),100]);
end;
J = zeros(100,100,1,100,'single');
for j = 1 : num - floor(num/100) * 100
    I = imread(C{1}{floor(num/100) * 100+j});   
    I = single(I') - meanC;
    J(:,:,1,j) = I / 128;
end;
H={J};
f = caffe('forward',H);
f=f{1};
f = reshape(f,[size(AllFeature,1),100]);
AllFeature(:,floor(num/100) * 100+1:num) = f(:,1 : num - floor(num/100) * 100);

AllFeature = bsxfun(@rdivide, AllFeature, sqrt(sum(AllFeature.^2)));
[normX, PCAmap] = compute_mapping(AllFeature', 'PCA', 310);
[mappedX, mapping] = JointBayesian(normX, label);

all_class = 300;
image_num_in_class = zeros(all_class,1);
image_in_class = cell(all_class,1);
thresh = zeros(length(C{2})/2,1);
for i = 1:all_class
    image_in_class{i} = find(C{2} == i -1);
    image_num_in_class(i) = length(image_in_class{i});
end;
for i=1:1000
    try
        [pair1,pair2] = getTrainList(C,image_in_class,image_num_in_class);
        break;
    catch
    end;
end;
for i=1 : length(pair1)
%     thresh(i) = normX(pair1(i),:) * mapping.A * normX(pair1(i),:)' + normX(pair2(i),:) * mapping.A * normX(pair2(i),:)' - 2 * normX(pair1(i),:) * mapping.G * normX(pair2(i),:)';
    thresh(i) = pdist2(AllFeature(:,pair1(i))',AllFeature(:,pair2(i))');
end;
sim_label = (C{2}(pair1)==C{2}(pair2));