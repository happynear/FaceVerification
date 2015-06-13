normX = AllFeature';
normX = bsxfun(@rdivide, normX, sqrt(sum(normX.^2,2)));
% [normX, PCAmap] = compute_mapping(normX, 'PCA', 319);
% fid = fopen('G:\face-CASIA-WebFace\sim_label_test.txt');
% C = textscan(fid,'%d');
% fclose(fid);
% sim = C{1};
% fid = fopen('G:\face-CASIA-WebFace\test_data1.txt');
% C = textscan(fid, '%s %d');
% fclose(fid);
% [mappedX, mapping] = JointBayesian(normX, C{2});
num = size(normX,1) / 2;


thresh = zeros(num,1);

% normX = AllFeature';
% normX = mappedX;
for i = 1:num
%     thresh(i) = normX(i,:) * mapping.A * normX(i,:)' + normX(num+i,:) * mapping.A * normX(num+i,:)' - 2 * normX(i,:) * mapping.G * normX(num+i,:)';
    thresh(i) = pdist2(normX(i,:), normX(num+i,:)); 
end;

num = size(AllFeature1,2);
F1 = AllFeature1';
F1 = bsxfun(@rdivide, F1, sqrt(sum(F1.^2,2)));
% F1 = bsxfun(@minus,F1,PCAmap.mean);
% F1 = F1 * PCAmap.M;
F2 = AllFeature2';
F2 = bsxfun(@rdivide, F2, sqrt(sum(F2.^2,2)));
% F2 = bsxfun(@minus,F2,PCAmap.mean);
% F2 = F2 * PCAmap.M;
% F1 = AllFeature1';
% F2 = AllFeature2';
thresh2 = zeros(num,1);
for i = 1:num
%     thresh2(i) = F1(i,:) * mapping.A * F1(i,:)' + F2(i,:) * mapping.A * F2(i,:)' - 2 * F1(i,:) * mapping.G * F2(i,:)';
    thresh2(i) = pdist2(F1(i,:),F2(i,:));
end;

% model = train(double(sim),sparse(thresh),'-s 2');
% predicted_label = predict(double(sim),sparse(thresh),model);
bestc=256;
lfw_label = ones(6000,1);
lfw_label(3001:6000) = 0;
% predicted_label = predict(double(lfw_label),sparse(thresh2),model);
cmd = [' -t 0 -h 0'];
model = svmtrain(double(sim),thresh,cmd);
[class] = svmpredict(lfw_label,thresh2,model);