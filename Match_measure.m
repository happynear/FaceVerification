function [accuracy, threshold]=Match_measure(filename)
fid=fopen(filename);
C=textscan(fid,'%s\t%s\t%d\t%d\t%f\n');
fclose(fid);
labels=C{4};
scores=C{5};
match=scores(labels==1);
nonmatch=scores(labels==0);
[accuracy,threshold] = Sys_accuracy(match, nonmatch);

