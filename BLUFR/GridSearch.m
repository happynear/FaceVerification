function [pcaDims, lambda] = GridSearch(trainX, trainY, testX, testY)
%% [pcaDims, lambda] = GridSearch(trainX, trainY, testX, testY)
%
% This function demostrates how to use the development set to learn the
% optimal parameters.
%
% Inputs:
%   trainX, trainY: training data and class labels.
%   testX, testY: test data and class labels.
% Outputs:
%   pcaDims: the optimal PCA dimensions.
%   lambda: the optimal ridge regression parameter.
% 
% Version: 1.0
% Date: 2014-07-22
%
% Author: Shengcai Liao
% Institute: National Laboratory of Pattern Recognition,
%   Institute of Automation, Chinese Academy of Sciences
% Email: scliao@nlpr.ia.ac.cn

candiPcaDims = 100:100:500; % candidate parameter values for the pca dimensions.
candiLambda = 10.^(-4:0); % candidate parameter values for the ridge regression.
veriFarPoints = [0, kron(10.^(-8:-1), 1:9), 1]; % FAR points for verification performance evaluation

fprintf('Learn parameters on the development set...\n');

%% Learn a PCA subspace.
W = PCA(trainX);

%% Transform both the training and test data into the learned PCA subspace.
trainX = trainX * W; 
testX = testX * W;

%% Select classes which have at least two images.
hst = hist(trainY, 1 : max(trainY));
classIndex = find(hst >= 2);
[sampleIndex, trainY] = ismember(trainY, classIndex); % class labels are continuously relabeled starting from 1
trainY = trainY(sampleIndex);
trainX = trainX(sampleIndex,:);
numTrainSamples = length(trainY);

%% Construct the discriminant response matrix
Y = zeros(numTrainSamples, length(classIndex));
Y( sub2ind(size(Y), 1 : numTrainSamples, trainY') ) = 1;

%% Pre-compute the working variables for the ridge regression.
R = trainX' * trainX;
Z = trainX' * Y;

numPara1 = length(candiPcaDims);
numPara2 = length(candiLambda);
auc = zeros(numPara1, numPara2);

%% Learn the optimal parameters by grid search
for i = 1 : numPara1
    d = candiPcaDims(i);
    
    for j = 1 : numPara2  
        % Learn a linear subspace by ridge regression. You can replace this with
        % your own supervised learning algorithm here.
        W = (R(1:d, 1:d) + candiLambda(j) * numTrainSamples * eye(d)) \ Z(1:d,:);
    
        % Transform the test data into the learned subspace.
        X = testX(:, 1:d) * W;

        % Normlize each row to unit length. If you do not have this function,
        % do it manually.
        X = normr(X);

        % Compute the cosine similarity score between the test samples.
        score = X * X';

        % Evaluate the verification performance.
        VR = EvalROC(score, testY, [], veriFarPoints);

        % Average the verification rates as the AUC performance
        auc(i,j) = mean(VR);
    end
end

%% Plot the AUC performance w.r.t. different PCA dimensions.
figure; semilogx(candiPcaDims, auc, 'LineWidth', 2);
grid on;
xlabel('PCA Dimension');
ylabel('AUC');
title('AUC performance w.r.t. different PCA dimensions');
labels = cellstr( [repmat('lambda = ', [length(candiLambda),1]), num2str(candiLambda')] );
legend(labels, 'Location', 'NorthWest');
set(gca, 'XTick', candiPcaDims);
set(gca, 'XTickLabel', cellstr(num2str(candiPcaDims')));
drawnow;

%% Get the best parameters.
[bestAuc, index] = max(auc(:));
[r,c] = ind2sub(size(auc), index);
pcaDims = candiPcaDims(r);
lambda = candiLambda(c);
fprintf('The best AUC: %g. The best PCA dimensions: %d. The best lambda: %g.\n\n', bestAuc, pcaDims, lambda);
