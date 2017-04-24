function [DIR, FAR, thresholds] = OpenSetROC( score, galLabels, probLabels, farPoints, rankPoints )
%% [DIR, FAR, thresholds] = OpenSetROC( score, galLabels, probLabels, farPoints, rankPoints )
% 
% A function for open-set identification ROC evaluation.
% 
% Inputs:
%     score: score matrix with rows corresponding to gallery and columns probe.
%     galLabels: a vector containing class labels of each gallery sample,
%           corresponding to rows of the score matrix.
%     probLabels: a vector containing class labels of each probe sample,
%           corresponding to columns of the score matrix.
%     farPoints: interested points of the false accept rates for
%           evaluation. Optional.
%     rankPoints: interested points of the matching ranks. Optional.
% 
% Outputs:
%     DIR: detection and identification rates, with rows corresponding to
%           ranks and columins corresponding to FARs
%     FAR: false accept rates. This is preferred than farPoints for
%       performance plot, because with insufficient number of impostor probes,
%       some small far points may cannot be reached, and the corresponding
%       returned FARs will be set to zero.
%     thresholds: decision thresholds used to generate DIR and FAR.
% 
% Example:
%     farPoints = [0, kron(10.^(-4:-1), 1:9), 1];
%     rankPoints = [1:10, 20:10:100];
%     [DIR, FAR, thresholds] = OpenSetROC( score, galLabels, probLabels, farPoints, rankPoints );
%     figure; surf(FAR * 100, rankPoints, DIR * 100);
%     xlabel( 'False Accept Rate (%)' );
%     ylabel( 'Rank' );
%     zlabel( 'Detection and Identification Rate (%)' );
%     title( 'Open-set Identification Performance' );
%     figure; semilogx( far*100, DIR(10,:)*100, 'r-o' ); grid on;
%     xlabel( 'False Accept Rate (%)' );
%     ylabel( 'Detection and Identification Rate (%)' );
%     title( 'Open-set Identification ROC Curve at Rank 10' );
%     [~, farIndex] = ismember(0.01, farPoints);
%     figure; semilogx( rankPoints, DIR(:,farIndex)*100, 'r-o' ); grid on;
%     xlabel( 'Rank' );
%     ylabel( 'Detection and Identification Rate (%)' );
%     title( 'Open-set Identification CMC Curve at FAR = 0.01' );
%
% Reference:
%   Shengcai Liao, Anil K. Jain, and Stan Z. Li, "Partial Face Recognition: Alignment Free Approach," 
%       IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(5):1193¨C1205, 2013.
% 
% Version: 1.0
% Date: 2014-07-22
%
% Author: Shengcai Liao
% Institute: National Laboratory of Pattern Recognition,
%   Institute of Automation, Chinese Academy of Sciences
% Email: scliao@nlpr.ia.ac.cn


%% preprocess
if nargin < 5 || isempty(rankPoints)
    rankPoints = [1:10, 20:10:100];
    rankPoints(rankPoints > length(galLabels)) = [];
end

if ~iscolumn(galLabels)
    galLabels = galLabels';
end

if ~isrow(probLabels)
    probLabels = probLabels';
end

binaryLabels = bsxfun(@eq, galLabels, probLabels); % match / non-match labels corresponding to the score matrix

t = any(binaryLabels == true, 1); % determine whether a probe belongs to the gallery
genProbIndex = find(t==true); % seperate the probe set into genuine probe set and impostor probe set
impProbIndex = find(t==false);
Ngen = length(genProbIndex);
Nimp = length(impProbIndex);

% set the number of false alarms
if nargin < 4 || isempty(farPoints)
    falseAlarms = 0 : Nimp;
else
    if any(farPoints < 0) || any(farPoints > 1)
        error('FAR should be in the range [0,1].');
    end
    falseAlarms = round(farPoints * Nimp);
end

%% get detection scores and matching ranks of each probe
impScore = max( score(:, impProbIndex) ); % maximum scores of each impostor probe
impScore = sort(impScore, 'descend');

S = score(:, genProbIndex); % matching scores of each genuine probe
[~, sortedIndex] = sort(S, 'descend'); % rank the score
M = binaryLabels(:, genProbIndex); % match / non-match labels
clear binaryLabels
S(M == false) = -Inf; % set scores of non-matches to -Inf
clear M
[genScore, genGalIndex] = max(S); % get maximum genuine score of the matches, as well as the location of the matches
clear S
[probRanks, ~] = find( bsxfun(@eq, sortedIndex, genGalIndex) ); % get the matching ranks of each genuine probe, by finding the location of the matches in the sorted index
clear sortedIndex

%% compute thresholds
isZeroFAR = (falseAlarms == 0);
isOneFAR = (falseAlarms == Nimp);
thresholds = zeros(1, length(falseAlarms));
thresholds(~isZeroFAR & ~isOneFAR) = impScore( falseAlarms(~isZeroFAR & ~isOneFAR) ); % use the sorted imporstor scores to generate the decision thresholds

% when FAR=0, the decision threshold should be a bit larger than
% impScore(1), because the decision is made by the ">=" operator
highGenScore = genScore(genScore > impScore(1));
if isempty(highGenScore)
    thresholds(isZeroFAR) = impScore(1) + sqrt(eps);
else
    thresholds(isZeroFAR) = ( impScore(1) + min(highGenScore) ) / 2;
end

% when FAR = 1, the decision threshold should be the minimum score that can
% also accept all genuine scores
thresholds(isOneFAR) = min(impScore(end), min(genScore)) - sqrt(eps);

%% evaluate
if ~iscolumn(genScore)
    genScore = genScore';
end

if ~isrow(thresholds)
    thresholds = thresholds';
end

if ~iscolumn(probRanks)
    probRanks = probRanks';
end

if ~isrow(rankPoints)
    rankPoints = rankPoints';
end

T1 = bsxfun(@ge, genScore, thresholds); % compare genuine scores to the decision thresholds
T2 = bsxfun(@le, probRanks, rankPoints); % compare the genuine probe matching ranks to the interested rank points
T = bsxfun(@and, reshape(T1, Ngen, 1, []), T2); % detection and identification should be both satisfied
DIR = squeeze( mean(T) ); % average over all genuine probes 
FAR = falseAlarms / Nimp;
