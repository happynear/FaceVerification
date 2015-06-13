load('E:\datasets\WDRef\id_lfw.mat');
load('E:\datasets\WDRef\imagelist_lfw.mat');
load('E:\datasets\WDRef\lbp_lfw.mat');
load('E:\datasets\WDRef\pairlist_lfw.mat');
% labels = double(train_y) * (0:9)';
labels = id_lfw;
X = double(lbp_lfw);
idx = [];

validate = 1;
tmp = pairlist_lfw.IntraPersonPair;
tmp((validate-1)*300+1:validate*300,:) = [];
idx = [idx;tmp(:)];
tmp = pairlist_lfw.ExtraPersonPair;
tmp((validate-1)*300+1:validate*300,:) = [];
idx = [idx;tmp(:)];
idx = unique(idx);
train_x = X(idx,:);
train_y = id_lfw(idx);
train_mean = mean(train_x, 1);
[COEFF,SCORE] = princomp(train_x,'econ');
train_x = SCORE(:,1:400);
normX = bsxfun(@minus,X,train_mean);
normX = normX * COEFF;
normX = normX(:,1:400);

[mappedX, mapping] = JointBayesian(train_x, train_y);
% [mappedX, mapping] = JointBayesian(X, labels);
% ss = X(1,:) * mapping.A * X(1,:)' + X(10,:) * mapping.A * X(10,:)' - 2 * X(1,:) * mapping.G * X(10,:)';
%     CD = u(:,7)' * COEFF(:,1:400)';
%     CD = CD + mapping.mean;
%     imshow(CD);
%     imshow(uint8(CD));
%     imshow(reshape(uint8(CD),28,28)');
% validate = 2;
validate = 1;
test_Intra = pairlist_lfw.IntraPersonPair((validate-1)*300+1:validate*300,:);
test_Extra = pairlist_lfw.ExtraPersonPair((validate-1)*300+1:validate*300,:);
result_Intra = zeros(300,1);
result_Extra = zeros(300,1);
for i=1:300
    result_Intra(i) = normX(test_Intra(i,1),:) * mapping.A * normX(test_Intra(i,1),:)' + normX(test_Intra(i,2),:) * mapping.A * normX(test_Intra(i,2),:)' - 2 * normX(test_Intra(i,1),:) * mapping.G * normX(test_Intra(i,2),:)';
    result_Extra(i) = normX(test_Extra(i,1),:) * mapping.A * normX(test_Extra(i,1),:)' + normX(test_Extra(i,2),:) * mapping.A * normX(test_Extra(i,2),:)' - 2 * normX(test_Extra(i,1),:) * mapping.G * normX(test_Extra(i,2),:)';
end;

% tmp = [pairlist_lfw.IntraPersonPair((validate-1)*300+1:validate*300,:);pairlist_lfw.ExtraPersonPair((validate-1)*300+1:validate*300,:)];
% idx = unique(tmp(:));

m = size(train_x,1);
    n = size(train_x,2);
	
	% Make sure labels are nice
	[classes, bar, train_y] = unique(train_y);
    nc = length(classes);
	
	% Intialize Sw
	Sw = zeros(size(train_x, 2), size(train_x, 2));
    
    cur = {};
    withinCount = 0;
    for i=1:nc
        % Get all instances with class i
        cur{i} = train_x(train_y == i,:);
        if size(cur{i},1)>1
            withinCount = withinCount + size(cur{i},1);
        end;
    end;

    u = zeros(n,nc);
	% Sum over classes
	for i=1:nc
		% Update within-class scatter
        u(:,i) = mean(cur{i},1)';
        if size(cur{i},1)>1
            C = cov(cur{i});
            p = size(cur{i}, 1) / (withinCount - 1);
            Sw = Sw + (p * C);
        end;
    end;
    Su = cov(u');
    F = inv(Sw);
G = -1 .* (2 .* Su + Sw) \ Su / Sw;
A = inv(Su + Sw) - (F + G);
c = zeros(m,1);
    for i = 1:m
        c(i) = train_x(i,:) * A * train_x(i,:)';
    end;
    result_Intra = zeros(300,1);
    result_Extra = zeros(300,1);
    for i=1:300
        result_Intra(i) = normX(test_Intra(i,1),:) * A * normX(test_Intra(i,1),:)' + normX(test_Intra(i,2),:) * A * normX(test_Intra(i,2),:)' - 2 * normX(test_Intra(i,1),:) * G * normX(test_Intra(i,2),:)';
        result_Extra(i) = normX(test_Extra(i,1),:) * A * normX(test_Extra(i,1),:)' + normX(test_Extra(i,2),:) * A * normX(test_Extra(i,2),:)' - 2 * normX(test_Extra(i,1),:) * G * normX(test_Extra(i,2),:)';
    end;
%     MM = X * G *X';
% DIS = repmat(c,1,length(c))+repmat(c,1,length(c))' - 2*MM;
% GT = zeros(m,m);
% for i = 1:m
%     for j=1:m
%         GT(i,j) = id_lfw(i)==id_lfw(j);
%     end;
% end;
% ER = (GT ~= (DIS>=0));
% totalER = sum(sum(abs(ER*DIS)));
% sumER = sum(sum(ER));