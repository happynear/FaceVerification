function [accuracy,threshold] = Sys_accuracy(match, nonmatch)
    [TAR,FAR,nonmatch_score] = rocplot(match, nonmatch);
    %[length(TAR),length(FAR),length(match),length(nonmatch_score),length(nonmatch)]
    accuracies=(TAR*length(match)+(1-FAR)*length(nonmatch))/(length(nonmatch)+length(match));
    [~,idx]=sort(abs(TAR-(1-FAR)),1,'ascend');

    [accuracy,a_idx]=max(accuracies(idx(1:100)));
    threshold=nonmatch_score(idx(a_idx(1)));