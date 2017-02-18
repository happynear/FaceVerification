fid = fopen('G:/100_25/zzzpairs.txt');
CC = fscanf(fid,'%d %d');
n_set = CC(1);n_num=CC(2);

same_pair = cell(n_set*n_num,2);
diff_pair = cell(n_set*n_num,2);
lfw_label = zeros(n_set*n_num * 2,2);
label_ht = java.util.Hashtable;

for i=1:n_set
    for j = 1 : n_num
        CC = textscan(fid, '%s %d %d\n');
        p = CC{1};id1=CC{2};id2=CC{3};
        same_pair((i-1)*n_num + j,1) = {sprintf('G:/100_25/%s_%04d-l.bmp',p{1},id1)};
        same_pair((i-1)*n_num + j,2) = {sprintf('G:/100_25/%s_%04d-l.bmp',p{1},id2)};
        if label_ht.containsKey(p{1})
            lfw_label((i-1)*n_num + j,1) = label_ht.get(p{1});
            lfw_label((i-1)*n_num + j,2) = label_ht.get(p{1});
        else
            idx = label_ht.size()+1;
            lfw_label((i-1)*n_num + j,1) = idx;
            lfw_label((i-1)*n_num + j,2) = idx;
            label_ht.put(p{1},idx);
        end;
    end;
    for j = 1 : n_num
         CC = textscan(fid, '%s %d %s %d\n');
         p1 = CC{1};id1=CC{2};p2=CC{3};id2=CC{4};
        diff_pair((i-1)*n_num + j,1) = {sprintf('G:/100_25/%s_%04d-l.bmp',p1{1},id1)};
        diff_pair((i-1)*n_num + j,2) = {sprintf('G:/100_25/%s_%04d-l.bmp',p2{1},id2)};
        if label_ht.containsKey(p1{1})
            lfw_label((i-1)*n_num + j + n_set*n_num,1) = label_ht.get(p1{1});
        else
            idx = label_ht.size()+1;
            label_ht.put(p1{1},idx);
            lfw_label((i-1)*n_num + j + n_set*n_num,1) = idx;
        end;
        if label_ht.containsKey(p2{1})
            lfw_label((i-1)*n_num + j + n_set*n_num,2) = label_ht.get(p2{1});
        else
            idx = label_ht.size()+1;
            label_ht.put(p2{1},idx);
            lfw_label((i-1)*n_num + j + n_set*n_num,2) = idx ;
        end;
    end;
end;
fclose(fid);