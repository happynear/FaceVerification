num = size(AllFeature1,2);
F1 = AllFeature1';
F1 = bsxfun(@rdivide, F1, sqrt(sum(F1.^2,2)));
F2 = AllFeature2';
F2 = bsxfun(@rdivide, F2, sqrt(sum(F2.^2,2)));

same_label = ones(6000,1);
same_label(3001:6000) = 0;

accuracies = zeros(10,1);

for i = 1 : 10
    test_idx = [(i-1) * 300 + 1 : i*300, (i-1) * 300 + 3001 : i*300 + 3000];
    train_idx = 1:6000;
    train_idx(test_idx) = [];
    train = [F1(train_idx,:);F2(train_idx,:)];
    train_label = [lfw_label(train_idx,1);lfw_label(train_idx,2)];
    [mappedX, mapping] = JointBayesian(train, train_label);
    
%     thresh = zeros(size(train_idx,2),1);
%     for j = 1:size(train_idx,2)
%         thresh(j) = F1(train_idx(j),:) * mapping.A * F1(train_idx(j),:)' + F2(train_idx(j),:) * mapping.A * F2(train_idx(j),:)' - 2 * F1(train_idx(j),:) * mapping.G * F2(train_idx(j),:)';
%     end;
    thresh = zeros(size(F1,1),1);
    for j = 1:size(F1,1)
        thresh(j) = F1(j,:) * mapping.A * F1(j,:)' + F2(j,:) * mapping.A * F2(j,:)' - 2 * F1(j,:) * mapping.G * F2(j,:)';
    end;
    cmd = [' -t 0 -h 0'];
    model = svmtrain(same_label(train_idx),thresh(train_idx),cmd);
    [class] = svmpredict(same_label(train_idx),thresh(train_idx),model);
    [class, accuracy, deci] = svmpredict(same_label(test_idx),thresh(test_idx),model);
    accuracies(i) = accuracy(1);
end;

mean(accuracies)