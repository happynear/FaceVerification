fid = fopen('G:\face-CASIA-WebFace\list.txt');
C = textscan(fid, '%s %d');
fclose(fid);
% C{1} = C{1}(1:2000);
% C{2} = C{2}(1:2000);
fid = fopen('G:\face-CASIA-WebFace\sim_label.txt');
C2 = textscan(fid, '%d');
% C2{1} = C2{1}(1:2000);
fclose(fid);
meanC = caffe('read_mean','D:\ThirdPartyLibrary\caffe\examples\siamese\mean.proto');

matcaffe_init(1,'D:\ThirdPartyLibrary\caffe\examples\siamese\mnist_siamese_deploy.prototxt','D:\ThirdPartyLibrary\caffe\examples\siamese\siamese_iter_86000.caffemodel');
num = length(C{2});
label = C2{1}(1:num/2);
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

thresh1 = [];
thresh2 = [];
p1=1;
p2=1;
num = floor(size(AllFeature,2) / 2);
normX = AllFeature';
for i = 1:num
    if C{2}(i) == C{2}(num+i)
        thresh1 = [thresh1;pdist2(normX(i,:), normX(num+i,:))];
        p1=p1+1;
    else
        thresh2 = [thresh2;pdist2(normX(i,:), normX(num+i,:))];
        p2=p2+1;
    end;
end;
mean(thresh1) / 4 + mean(max(100 - thresh2, 0)) / 4