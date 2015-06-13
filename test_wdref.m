load('D:\wf\face\dataset\WDRef\lbp_WDRef.mat')
load('D:\wf\face\dataset\WDRef\id_WDRef.mat')
labels = id_WDRef;
X = double(lbp_WDRef);
X = sqrt(X);
X = bsxfun(@rdivide,X,sum(X,2));
train_mean = mean(X, 1);
[COEFF,SCORE] = princomp(X,'econ');
train_x = SCORE(:,1:1000);

[mappedX, mapping] = JointBayesian(train_x, labels);
% Dis_matrix = repmat(mapping.c,1,size(train_x,1))+repmat(mapping.c,1,size(train_x,1))+train_x * mapping.G *train_x';
[classes, bar, labels] = unique(labels);
    nc = length(classes);
train_Intra = zeros(nc*2,2);
for i=1:nc
    train_Intra(2*i-1,:) = randperm(sum(labels == i),2) + find(labels == i,1,'first') - 1;
    train_Intra(2*i,:) = randperm(sum(labels == i),2) + find(labels == i,1,'first') - 1;
end;
train_Extra = reshape(randperm(length(labels),20000),10000,2);
train_Extra(labels(train_Extra(:,1))==labels(train_Extra(:,2)),:)=[];
train_Extra(size(train_Intra,1)+1:end,:)=[];
Dis_train_Intra = zeros(size(train_Intra,1),1);
Dis_train_Extra = zeros(size(train_Intra,1),1);

for i=1:size(train_Intra,1)
    Dis_train_Intra(i) = train_x(train_Intra(i,1),:) * mapping.A * train_x(train_Intra(i,1),:)' + train_x(train_Intra(i,2),:) * mapping.A * train_x(train_Intra(i,2),:)' - 2 * train_x(train_Intra(i,1),:) * mapping.G * train_x(train_Intra(i,2),:)';
    Dis_train_Extra(i) = train_x(train_Extra(i,1),:) * mapping.A * train_x(train_Extra(i,1),:)' + train_x(train_Extra(i,2),:) * mapping.A * train_x(train_Extra(i,2),:)' - 2 * train_x(train_Extra(i,1),:) * mapping.G * train_x(train_Extra(i,2),:)';
end;
group_train = [ones(size(Dis_train_Intra,1),1);zeros(size(Dis_train_Extra,1),1)];
training = [Dis_train_Intra;Dis_train_Extra];

load('D:\wf\face\dataset\WDRef\lbp_lfw.mat')
load('D:\wf\face\dataset\WDRef\pairlist_lfw.mat')
normX = double(lbp_lfw);
normX = sqrt(normX);
normX = bsxfun(@rdivide,normX,sum(normX,2));
normX = bsxfun(@minus,normX,train_mean);
normX = normX * COEFF(:,1:1000);
test_Intra = pairlist_lfw.IntraPersonPair;
test_Extra = pairlist_lfw.ExtraPersonPair;

result_Intra = zeros(3000,1);
result_Extra = zeros(3000,1);
for i=1:3000
    result_Intra(i) = normX(test_Intra(i,1),:) * mapping.A * normX(test_Intra(i,1),:)' + normX(test_Intra(i,2),:) * mapping.A * normX(test_Intra(i,2),:)' - 2 * normX(test_Intra(i,1),:) * mapping.G * normX(test_Intra(i,2),:)';
    result_Extra(i) = normX(test_Extra(i,1),:) * mapping.A * normX(test_Extra(i,1),:)' + normX(test_Extra(i,2),:) * mapping.A * normX(test_Extra(i,2),:)' - 2 * normX(test_Extra(i,1),:) * mapping.G * normX(test_Extra(i,2),:)';
end;

group_sample = [ones(3000,1);zeros(3000,1)];
sample = [result_Intra;result_Extra];

bestc=256;
% bestg=128;
% cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg)];
cmd = [' -t 0 -h 0'];
model = svmtrain(group_train,training,cmd);
[class,accTotal] = svmpredict(group_sample,sample,model);

% (sum(result_Intra>0)+sum(result_Extra<0))/6000

% mapping.G = -1 .* (1.5 * mapping.Su + mapping.Sw) \ mapping.Su / mapping.Sw;
% mapping.A = inv(mapping.Su + mapping.Sw) - (inv(mapping.Sw) + mapping.G);
% for i=1:3000
%     result_Intra(i) = normX(test_Intra(i,1),:) * mapping.A * normX(test_Intra(i,1),:)' + normX(test_Intra(i,2),:) * mapping.A * normX(test_Intra(i,2),:)' - 2 * normX(test_Intra(i,1),:) * mapping.G * normX(test_Intra(i,2),:)';
%     result_Extra(i) = normX(test_Extra(i,1),:) * mapping.A * normX(test_Extra(i,1),:)' + normX(test_Extra(i,2),:) * mapping.A * normX(test_Extra(i,2),:)' - 2 * normX(test_Extra(i,1),:) * mapping.G * normX(test_Extra(i,2),:)';
% end;
% (sum(result_Intra>0)+sum(result_Extra<0))/6000