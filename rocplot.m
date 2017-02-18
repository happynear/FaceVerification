function [TAR,FAR,threshold] = rocplot(match, nonmatch)
%TAR=Ver

%match=-match;
%nonmatch=-nonmatch;

 R1=min([match; nonmatch]);
 R2=max([nonmatch;match]);

a=100/(R2-R1);
b=-a*R1;

matchN=a*match+b;
matchN(matchN<0)=0;
nonmatch_score=a*nonmatch+b;
TAR=[1:length(matchN)]'/length(matchN);
FRR=1-TAR;

nonmatchL=zeros(size(matchN));
smatch=sort(matchN);
for i=1:length(nonmatch_score)
    low=1;
    high=length(smatch);
	while low <=high
		mid=ceil((low+high)*.5);
%[nonmatch_score(i),smatch(mid)]
		if nonmatch_score(i)>smatch(mid) low=mid+1;
		else high=mid-1;
		end
	end

 %   [val,idx]=min((nonmatch_score(i)>smatch));
 %    nonmatchL(idx)=nonmatchL(idx)+1;
	if low > length(smatch) 
	low= length(smatch);
	end
    nonmatchL(low)=nonmatchL(low)+1;
end
cnonmatchL=cumsum(nonmatchL);
FAR=cnonmatchL/cnonmatchL(end);
iind=find(FAR>=0.1);
VER1= TAR(iind(1));
iind=find(FAR>=0.01);
VER01= TAR(iind(1));
iind=find(FAR>=0.001);
VER001= TAR(iind(1));

threshold=(smatch-b)/a;


