fid = fopen('list.txt');
C = textscan(fid, '%s %d');
all_class = 10575;
image_num_in_class = zeros(all_class,1);
image_in_class = cell(all_class,1);
for i = 1:all_class
    image_in_class{i} = find(C{2} == i -1);
    image_num_in_class(i) = length(image_in_class{i});
end;
fclose(fid);
% clear all;
% load('temp1.mat');

test_class = 2000-1;
test_num = sum(C{2}>all_class-test_class);
total = length(C{2});
train_num = total - test_num;
same_r = 0.5;

same_pair_l = floor(test_num / 2 * same_r);
same_in_class = randi(test_class,floor(same_pair_l * 2),1) + all_class-test_class;
same_in_class_unique = unique(same_in_class);
got_in_same = ones(total,1);
got_in_same(train_num+1:end) = 0;
pair1_same = zeros(same_pair_l * 10,1);
pair2_same = zeros(same_pair_l * 10,1);
p=1;
for i = 1: length(same_in_class_unique)
    if mod(p,same_pair_l/100) == 0
        disp([p same_pair_l]);
    end;
    c = same_in_class_unique(i);
    n = sum(same_in_class == same_in_class_unique(i)) * 2;
    s = image_num_in_class(c);
    if n > s
        n = floor(s / 2) * 2;
    end;
    idx_same = randperm(s);
    for j = 1 : n / 2
        pair1_same(p) = image_in_class{c}(idx_same(j*2-1));
        pair2_same(p) = image_in_class{c}(idx_same(j*2));
%         got_in_same(image_in_class{c}(idx_same(j*2-1))) = 1;
%         got_in_same(image_in_class{c}(idx_same(j*2))) = 1;
%         if p==same_pair_l
%             break;
%         end;
        p = p + 1;
    end;
end;
assert(p>same_pair_l);
idx = randperm(p-1);
pair1_same = pair1_same(idx(1:same_pair_l));
pair2_same = pair2_same(idx(1:same_pair_l));
assert(sum(C{2}(pair1_same) == C{2}(pair2_same)) == same_pair_l);
got_in_same(pair1_same) = 1;
got_in_same(pair2_same) = 1;
diff_num = test_num - length(pair1_same) * 2;
diff_sample = find(got_in_same==0);
idx = randperm(diff_num);
diff_sample = diff_sample(idx);
pair1_diff = diff_sample(1:(diff_num/2));
pair2_diff = diff_sample(diff_num/2 + 1:end);
err = find(C{2}(pair1_diff) == C{2}(pair2_diff));
% for i = 1 : length(err)
    ridx = randperm(length(err));
    pair1_diff(err) = pair1_diff(err(ridx));
% end;
assert(sum(C{2}(pair1_diff) == C{2}(pair2_diff)) == 0);
pair1 = [pair1_same;pair1_diff];
pair2 = [pair2_same;pair2_diff];
label_sim = [ones(length(pair1_same),1);zeros(length(pair1_diff),1)];
idx = randperm(length(pair1));
pair1 = pair1(idx);
pair2 = pair2(idx);
label_sim = label_sim(idx);
assert(sum((C{2}(pair1) == C{2}(pair2)) ~= label_sim) == 0);
fid1 = fopen('pair_data1_test.txt','w');
fid2 = fopen('pair_data2_test.txt','w');
fid3 = fopen('sim_label_test.txt','w');
for i = 1 : length(pair1)
    fprintf(fid1,'%s %d\r\n',C{1}{pair1(i)},C{2}(pair1(i)));
    fprintf(fid2,'%s %d\r\n',C{1}{pair2(i)},C{2}(pair2(i)));
    fprintf(fid3,'%d\r\n',label_sim(i));
end;
fclose(fid1);
fclose(fid2);
fclose(fid3);