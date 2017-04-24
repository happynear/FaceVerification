function [W, S, latent] = PCA(X, pcaDims)
%% [W, S, latent] = PCA(X, pcaDims)
% 
% Function for learning a Principle Component Analysis (PCA) subspace.
% 
% Inputs:
%   X: training data, with rows corresponding to samples and columns features.
%   pcaDims: optional parameter specifying the output PCA subspace
%       dimensions.
% 
% Outputs:
%   W: the learned PCA subspace projection matrix.
%   S: projected coefficients of X by W.
%   latent: latent values of the covariant matrix
% 
% Version: 1.0
% Date: 2014-07-22
%
% Author: Shengcai Liao
% Institute: National Laboratory of Pattern Recognition,
%   Institute of Automation, Chinese Academy of Sciences
% Email: scliao@nlpr.ia.ac.cn

[n, d] = size(X);

if nargin == 1
    pcaDims = min(d,n) - 1;
end

if pcaDims <= 0 || pcaDims >= min(d,n)
    error('Incorrect pcaDims.');
end

fprintf('Start to train PCA...');
t0 = tic;

mu = mean(X);
X = bsxfun(@minus, X, mu);

if n >= d
    C = X' * X; %[d,d]
else
    C = X * X'; %[n,n]
end

[W, D] = eig(C);
latent = diag(D);
[latent, index] = sort(latent, 'descend');
W = W(:, index);

W = W(:, 1:pcaDims);
latent = latent(1:pcaDims);

if n < d
    W = X' * W * diag(1 ./ sqrt(latent));
end

if nargout >= 2 && ~isempty(S)
    S = X * W;
end

fprintf('%.3g seconds.\n', toc(t0));
