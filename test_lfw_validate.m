load('D:\wf\face\dataset\WDRef\id_lfw.mat');
load('D:\wf\face\dataset\WDRef\imagelist_lfw.mat');
load('D:\wf\face\dataset\WDRef\lbp_lfw.mat');
load('D:\wf\face\dataset\WDRef\pairlist_lfw.mat');
% labels = double(train_y) * (0:9)';
labels = id_lfw;
X = double(lbp_lfw_baseline_cvpr12);
% X = sqrt(X);
% X = bsxfun(@rdivide,X,sum(X.^2,2));
idx = [];

validate = 9;
testing = 10;
tmp = pairlist_lfw.IntraPersonPair;
tmp([(validate-1)*300+1:validate*300;(testing-1)*300+1:testing*300],:) = [];
train_Intra = tmp;
idx = [idx;tmp(:)];
tmp = pairlist_lfw.ExtraPersonPair;
tmp([(validate-1)*300+1:validate*300;(testing-1)*300+1:testing*300],:) = [];
train_Extra = tmp;
idx = [idx;tmp(:)];
idx = unique(idx);
train_x = X(idx,:);
train_y = id_lfw(idx);
train_mean = mean(train_x, 1);
[COEFF,SCORE] = princomp(train_x,'econ');
train_x = SCORE(:,1:1000);
normX = bsxfun(@minus,X,train_mean);
normX = normX * COEFF;
normX = normX(:,1:1000);

[mappedX, mapping] = JointBayesian(train_x, train_y);

validate_Intra = pairlist_lfw.IntraPersonPair((validate-1)*300+1:validate*300,:);
validate_Extra = pairlist_lfw.ExtraPersonPair((validate-1)*300+1:validate*300,:);
Dis_validate_Intra = zeros(300,1);
Dis_validate_Extra = zeros(300,1);

for i=1:300
    Dis_validate_Intra(i) = normX(validate_Intra(i,1),:) * mapping.A * normX(validate_Intra(i,1),:)' + normX(validate_Intra(i,2),:) * mapping.A * normX(validate_Intra(i,2),:)' - 2 * normX(validate_Intra(i,1),:) * mapping.G * normX(validate_Intra(i,2),:)';
    Dis_validate_Extra(i) = normX(validate_Extra(i,1),:) * mapping.A * normX(validate_Extra(i,1),:)' + normX(validate_Extra(i,2),:) * mapping.A * normX(validate_Extra(i,2),:)' - 2 * normX(validate_Extra(i,1),:) * mapping.G * normX(validate_Extra(i,2),:)';
end;
group_train = [ones(2700,1);zeros(2700,1)];
training = [Dis_validate_Intra;Dis_validate_Extra];
% thresh1 = min(min(Dis_validate_Intra),max(Dis_validate_Extra));
% thresh2 = max(min(Dis_validate_Intra),max(Dis_validate_Extra));
% 
% CrossData = [Dis_validate_Intra(Dis_validate_Intra>=thresh1&Dis_validate_Intra<=thresh2);Dis_validate_Extra(Dis_validate_Extra>=thresh1&Dis_validate_Extra<=thresh2)];
% thresh = mean(CrossData);

test_Intra = pairlist_lfw.IntraPersonPair((testing-1)*300+1:testing*300,:);
test_Extra = pairlist_lfw.ExtraPersonPair((testing-1)*300+1:testing*300,:);
result_Intra = zeros(300,1);
result_Extra = zeros(300,1);

for i=1:300
    result_Intra(i) = normX(test_Intra(i,1),:) * mapping.A * normX(test_Intra(i,1),:)' + normX(test_Intra(i,2),:) * mapping.A * normX(test_Intra(i,2),:)' - 2 * normX(test_Intra(i,1),:) * mapping.G * normX(test_Intra(i,2),:)';
    result_Extra(i) = normX(test_Extra(i,1),:) * mapping.A * normX(test_Extra(i,1),:)' + normX(test_Extra(i,2),:) * mapping.A * normX(test_Extra(i,2),:)' - 2 * normX(test_Extra(i,1),:) * mapping.G * normX(test_Extra(i,2),:)';
end;
group_sample = [ones(300,1);zeros(300,1)];
sample = [result_Intra;result_Extra];

bestc=256;
% bestg=128;
% cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg)];
cmd = ['-c ', num2str(bestc), ' -t 0 '];
model = svmtrain(group_train,training,cmd);
[class,accTotal] = svmpredict(group_sample,sample,model);


(sum(result_Intra>thresh)+sum(result_Extra<thresh))/600
sum(result_Intra>0)
sum(result_Extra<0)

% thresh1 = min(min(result_Intra),max(result_Extra));
% thresh2 = max(min(result_Intra),max(result_Extra));
% 
% CrossData = [result_Intra(result_Intra>=thresh1&result_Intra<=thresh2);result_Extra(result_Extra>=thresh1&result_Extra<=thresh2)];
% thresh = mean(CrossData);


% for i=1:300
%     result_Intra(i) = -thresh - 2 * normX(test_Intra(i,1),:) * mapping.G * normX(test_Intra(i,2),:)';
%     result_Extra(i) = -thresh - 2 * normX(test_Extra(i,1),:) * mapping.G * normX(test_Extra(i,2),:)';
% end;
% (sum(result_Intra>0)+sum(result_Extra<0))/600
% 
% 
% 
% % % mapping.Sw = mapping.Sw *  4479 / 6916;
% % mapping.G = -1 .* (2 * mapping.Su + mapping.Sw) \ mapping.Su / mapping.Sw;
% % mapping.A = inv(mapping.Su + mapping.Sw) - (inv(mapping.Sw) + mapping.G);
% %  HI = [mapping.Su + mapping.Sw mapping.Su;mapping.Su mapping.Su + mapping.Sw];
% % HE = [mapping.Su + mapping.Sw zeros(size(mapping.Sw));zeros(size(mapping.Sw)) mapping.Su + mapping.Sw];
% % IHI = inv(HI);
% % IHE = inv(HE);
% % 
% % for i=1:300
% %     result_Intra(i) = [normX(test_Intra(i,1),:) normX(test_Intra(i,2),:)] * IHE *[normX(test_Intra(i,1),:) normX(test_Intra(i,2),:)]' - [normX(test_Intra(i,1),:) normX(test_Intra(i,2),:)] * IHI *[normX(test_Intra(i,1),:) normX(test_Intra(i,2),:)]';
% %     result_Extra(i) = [normX(test_Extra(i,1),:) normX(test_Extra(i,2),:)] * IHE *[normX(test_Extra(i,1),:) normX(test_Extra(i,2),:)]' - [normX(test_Extra(i,1),:) normX(test_Extra(i,2),:)] * IHI *[normX(test_Extra(i,1),:) normX(test_Extra(i,2),:)]';
% % end;
% % sum(result_Intra>0)
% % sum(result_Extra<0)
% 
% % tmp = [pairlist_lfw.IntraPersonPair((validate-1)*300+1:validate*300,:);pairlist_lfw.ExtraPersonPair((validate-1)*300+1:validate*300,:)];
% % idx = unique(tmp(:));
% 
%     m = size(train_x,1);
%     n = size(train_x,2);
% 	
% 	% Make sure labels are nice
% 	[classes, bar, train_y] = unique(train_y);
%     nc = length(classes);
% 	
% 	% Intialize Sw
% 	Sw = zeros(size(train_x, 2), size(train_x, 2));  
%     
%     cur = {};
%     withinCount = 0;
%     for i=1:nc
%         % Get all instances with class i
%         cur{i} = train_x(train_y == i,:);
%         if size(cur{i},1)>1
%             withinCount = withinCount + size(cur{i},1);
%         end;
%     end;
% 
%     u = zeros(n,nc);
% 	% Sum over classes
% 	for i=1:nc
% 		% Update within-class scatter
%         u(:,i) = mean(cur{i},1)';
%         if size(cur{i},1)>1
%             C = cov(cur{i});
%             p = size(cur{i}, 1) / (withinCount - 1);
%             Sw = Sw + (p * C);
%         end;
%     end;
%     Su = cov(u');
%     F = inv(Sw);
%     G = -1 .* (2 .* Su + Sw) \ Su / Sw;
%     A = inv(Su + Sw) - (F + G);
%     c = zeros(m,1);
%     for i = 1:m
%         c(i) = train_x(i,:) * A * train_x(i,:)';
%     end;
%     result_Intra = zeros(300,1);
%     result_Extra = zeros(300,1);
%     for i=1:300
%         result_Intra(i) = normX(test_Intra(i,1),:) * A * normX(test_Intra(i,1),:)' + normX(test_Intra(i,2),:) * A * normX(test_Intra(i,2),:)' - 2 * normX(test_Intra(i,1),:) * G * normX(test_Intra(i,2),:)';
%         result_Extra(i) = normX(test_Extra(i,1),:) * A * normX(test_Extra(i,1),:)' + normX(test_Extra(i,2),:) * A * normX(test_Extra(i,2),:)' - 2 * normX(test_Extra(i,1),:) * G * normX(test_Extra(i,2),:)';
%     end;
% %     MM = X * G *X';
% % DIS = repmat(c,1,length(c))+repmat(c,1,length(c))' - 2*MM;
% % GT = zeros(m,m);
% % for i = 1:m
% %     for j=1:m
% %         GT(i,j) = id_lfw(i)==id_lfw(j);
% %     end;
% % end;
% % ER = (GT ~= (DIS>=0));
% % totalER = sum(sum(abs(ER*DIS)));
% % sumER = sum(sum(ER));