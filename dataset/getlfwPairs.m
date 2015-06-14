fid = fopen('pairs.txt');
CC = fscanf(fid,'%d %d');
n_set = CC(1);n_num=CC(2);

same_pair = cell(n_set*n_num,2);
diff_pair = cell(n_set*n_num,2);
lfw_label = zeros(n_set*n_num * 2,2);

for i=1:n_set
    for j = 1 : n_num
        CC = textscan(fid, '%s %d %d\n');
        p = CC{1};id1=CC{2};id2=CC{3};
        same_pair((i-1)*n_num + j,1) = {sprintf('G:/face-lfw/%s/%s_%04d.jpg',p{1},p{1},id1)};
        same_pair((i-1)*n_num + j,2) = {sprintf('G:/face-lfw/%s/%s_%04d.jpg',p{1},p{1},id2)};
        lfw_label(j,1) = id1;
        lfw_label(j,2) = id2;
    end;
    for j = 1 : n_num
         CC = textscan(fid, '%s %d %s %d\n');
         p1 = CC{1};id1=CC{2};p2=CC{3};id2=CC{4};
        diff_pair((i-1)*n_num + j,1) = {sprintf('G:/face-lfw/%s/%s_%04d.jpg',p1{1},p1{1},id1)};
        diff_pair((i-1)*n_num + j,2) = {sprintf('G:/face-lfw/%s/%s_%04d.jpg',p2{1},p2{1},id2)};
        lfw_label(n_num+j,1) = id1;
        lfw_label(n_num+j,2) = id2;
    end;
end;
fclose(fid);