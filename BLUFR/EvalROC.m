function [VR, FAR, thresholds] = EvalROC(score, galLabels, probLabels, farPoints)
%% function [VR, FAR, thresholds] = EvalROC(score, galLabels, probLabels, farPoints)
%
% This function computes the verification rates and the false accept rates
% given the matching scores as well as the class labels associated with the gallery and probe images. 
%
% Inputs:
%   score: a matrix of matching scores for the gallery and probe images, with rows 
%       corresponding to gallery images and columns corresponding to probe images.
%   galLabels: class labels for the gallery images.
%   probLabels: class labels for the probe images. If this is empty, the
%       function will treat the matching scores as a square matrix with
%       probLabels=galLabels.
%   farPoints: optional FAR points used to evaluate the verification rates at particular FARs.
%
% Outputs:
% 	VR: verification rates (row vector).
% 	FAR: false accept rates (row vector). This is preferred than farPoints for
%     performance plot, because with insufficient number of impostor probes,
%     some small far points may cannot be reached, and the corresponding
%     returned FARs will be set to zero.
%   thresholds: decision thresholds corresponding to the VR and FAR.
%
% Version: 1.0
% Date: 2014-07-22
%
% Author: Shengcai Liao
% Institute: National Laboratory of Pattern Recognition,
%   Institute of Automation, Chinese Academy of Sciences
% Email: scliao@nlpr.ia.ac.cn

if nargin < 3 || isempty(probLabels)
    probLabels = galLabels;
    scoreMask = tril(true(size(score)), -1);
end

if ~iscolumn(galLabels)
    galLabels = galLabels';
end

if ~isrow(probLabels)
    probLabels = probLabels';
end

binaryLabels = bsxfun(@eq, galLabels, probLabels);

if ~isequal(size(score), size(binaryLabels))
    error('The size of labels is not the same as the size of the score matrix.');
end

if exist('scoreMask', 'var')
    score = score(scoreMask);
    binaryLabels = binaryLabels(scoreMask);
end

genScore = score(binaryLabels == true);
impScore = score(binaryLabels == false);
clear score binaryLabels

Nimp = length(impScore);

if nargin < 4 || isempty(farPoints)
    falseAlarms = 0 : Nimp;
else
    if any(farPoints < 0) || any(farPoints > 1)
        error('FAR should be in the range [0,1].');
    end
    falseAlarms = round(farPoints * Nimp);
end

impScore = sort(impScore, 'descend');

isZeroFAR = (falseAlarms == 0);
isOneFAR = (falseAlarms == Nimp);
thresholds = zeros(1, length(falseAlarms));
thresholds(~isZeroFAR & ~isOneFAR) = impScore( falseAlarms(~isZeroFAR & ~isOneFAR) );

highGenScore = genScore(genScore > impScore(1));
if isempty(highGenScore)
    thresholds(isZeroFAR) = impScore(1) + sqrt(eps);
else
    thresholds(isZeroFAR) = ( impScore(1) + min(highGenScore) ) / 2;
end

thresholds(isOneFAR) = min(impScore(end), min(genScore)) - sqrt(eps);

if ~iscolumn(genScore)
    genScore = genScore';
end

if ~isrow(thresholds)
    thresholds = thresholds';
end

FAR = falseAlarms / Nimp;
VR = mean( bsxfun(@ge, genScore, thresholds) );
