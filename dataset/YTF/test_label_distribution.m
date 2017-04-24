accuracies = zeros(10,1);
distance_kernel = @(A,B) squeeze(sum(bsxfun(@min,A,permute(B,[3 2 1])),2));%histogram intersection
% opts = [];
% opts.loss = 'exploss'; % can be logloss or exploss
% % gradient boost options
% opts.shrinkageFactor = 0.1;
% opts.subsamplingFactor = 0.5;
% opts.maxTreeDepth = uint32(2);  % this was the default before customization
% opts.randSeed = uint32(rand()*1000);
decisions = zeros(length(C{1}),1);
for cross = 1:10
    distance_cross = distance_dist{cross};
    cmd = [' -t 4 -h 0'];
    train_feature = distance_cross(C{1}~=cross,:);
    test_feature = distance_cross(C{1}==cross,:);
    model = svmtrain(double(C{5}(C{1}~=cross)), [(1:4500)' distance_kernel(train_feature,train_feature)],cmd);
    [class, accuracy, deci] = svmpredict(double(C{5}(C{1}==cross)),[(1:500)' distance_kernel(test_feature,train_feature)],model);
%     model = SQBMatrixTrain(single(train_feature), (double(C{5}(C{1}~=cross)) - 0.5)*2,uint32(10000),opts);
%     pred = SQBMatrixPredict(model, single(test_feature));
%     accuracy(1) = sum((pred>0) == C{5}(C{1}==cross)) / length(pred);
    fprintf('%d th-fold accuracy:%.4f\n', cross, accuracy(1));
    accuracies(cross) = accuracy(1);
    decisions((cross-1)*500+1:cross*500) = deci;
end;
fprintf('accuracy by 10-fold evalution:%.4f\n', mean(accuracies));
figure(1);
roc_curve(decisions, C{5}*2-1);
figure(2);
imagesc('XData',1:200,'YData',-1:0.01:1,'CData',test_feature(151:350,:)');
axis([0, 200, -1,1]);
colormap(parula);
colorbar;
caxis([0 0.15]);%'auto'
xlabel('video pair');
ylabel('face similarity');