close all;
num = size(AllFeature1,2);
F1 = AllFeature1' / 4000;
% F1 = max(AllFeature1(1:512,:)', AllFeature1(513:1024,:)');%(1:512,:)
% F1 = bsxfun(@minus, F1, mean(F1,1));
% F1 = bsxfun(@rdivide, F1, sqrt(sum(F1.^2,2)));

F2 = AllFeature2' / 4000;
% F2 = max(AllFeature2(1:512,:)', AllFeature2(513:1024,:)');%(1:512,:)
% F2 = bsxfun(@minus, F2, mean(F2,1));
% F2 = bsxfun(@rdivide, F2, sqrt(sum(F2.^2,2)));

thresh2 = zeros(num,1);
for i = 1:num
%     thresh2(i) = F1(i,:) * mapping.A * F1(i,:)' + F2(i,:) * mapping.A * F2(i,:)' - 2 * F1(i,:) * mapping.G * F2(i,:)';
    thresh2(i) = pdist2(F1(i,:) ./ norm(F1(i,:)),F2(i,:) ./ norm(F2(i,:)));
%     thresh2(i) = F1(i,:) * F2(i,:)';
end;
figure;
hist(thresh2(1:3000),500);
figure;
hist(thresh2(3001:end),500);

accuracies = zeros(10,1);
for i=1:10
    test_idx = [(i-1) * 300 + 1 : i*300, (i-1) * 300 + 3001 : i*300 + 3000];
    train_idx = 1:6000;
    train_idx(test_idx) = [];
    bestc=256;
    same_label = ones(6000,1);
    same_label(3001:6000) = 0;
    
    mean_feature = mean([F1(train_idx,:); F2(train_idx,:)]);
    std_feature = std([F1(train_idx,:); F2(train_idx,:)]);
    F1_mu = bsxfun(@minus, F1, mean_feature);
    F2_mu = bsxfun(@minus, F2, mean_feature);
%     F1_mu = bsxfun(@rdivide, F1_mu, std_feature);
%     F2_mu = bsxfun(@rdivide, F2_mu, std_feature);
    F1_mu = bsxfun(@rdivide, F1_mu, sqrt(sum(F1_mu.^2,2)));
    F2_mu = bsxfun(@rdivide, F2_mu, sqrt(sum(F2_mu.^2,2)));
    F1_mu = F1;
    F2_mu = F2;

    [U,mu,vars] = pca( [F1_mu(train_idx,:); F2_mu(train_idx,:)]' );
    sum_var = cumsum(vars);
    sum_var = sum_var / sum_var(end);
    dims = find(sum_var > 0.995, 1, 'first');
    [F1PCA,F1PCAHat,~] = pcaApply( F1_mu', U, mu, dims );
    [F2PCA,F2PCAHat,~] = pcaApply( F2_mu', U, mu, dims );
    F1PCA = bsxfun(@rdivide, F1PCA, sqrt(sum(F1PCA.^2)));
    F2PCA = bsxfun(@rdivide, F2PCA, sqrt(sum(F2PCA.^2)));
    
    thresh1 = zeros(num,1);
    for n = 1:num
%         thresh1(n) = pdist2(F1_mu(n,:),F2_mu(n,:));
%         thresh1(n) = pdist2(F1(n,:),F2(n,:));
        thresh1(n) = pdist2(F1PCA(:,n)',F2PCA(:,n)');
%         thresh1(n) = F1PCA(:,n)' * F2PCA(:,n);
%         thresh1(n) = F1(n,:) * F2(n,:)';
    end;
    cmd = [' -t 0 -h 0'];
    model = svmtrain(same_label(train_idx),thresh1(train_idx),cmd);
    [class, accuracy, deci] = svmpredict(same_label(test_idx),thresh1(test_idx),model);
%     [~, threshold_fold] = Sys_accuracy(thresh1(train_idx(1:2700)), thresh1(train_idx(2701:5400)));
%     accuracy = (sum(thresh1(test_idx(1:300)) < threshold_fold) + sum(thresh1(test_idx(301:600)) >= threshold_fold)) / 600;
    fprintf('%d th-fold accuracy:%.4f\n', i, accuracy(1));
    accuracies(i) = accuracy(1);
end;
fprintf('(PCA)accuracy by 10-fold evalution:%.4f\r\n', mean(accuracies));
cmd = [' -t 0 -h 0'];
model = svmtrain(same_label,thresh2,cmd);
[class, accuracy, deci] = svmpredict(same_label,thresh2,model);
% mean(thresh2(same_label==1)) / 4 + mean(max(0,1 - thresh2(same_label==0))) / 4
% sum((thresh2<0.22) == same_label) / 6000
% [accuracy, threshold] = Sys_accuracy(thresh2(same_label==1), thresh2(same_label==0));
% fprintf('(Without PCA)accuracy by direct evalution:%.4f, threshold:%.4f\r\n', accuracy, threshold);