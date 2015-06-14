function [pair1,pair2] = getTrainList(C,image_in_class,image_num_in_class)
% A quite dirty implementation of generating same face pair and different
% face pair. The ratio of same/diff can be controlled by same_r. If this
% code do wrong, just re-run it. Please refer to get10TrainList.m for
% usage.
all_class = 10575;
% clear all;
% load('temp1.mat');

test_class = 0;
test_num = sum(C{2}>all_class-test_class);
total = length(C{2});
train_num = total - test_num;
same_r = 0.5;
% idx = randperm(train_num);
% pair1 = idx(1:train_num/2);
% pair2 = idx(train_num/2+1:end);
% pair1_same = pair1(1:floor(train_num/2 *same_r));
% pair1_same_l = length(pair1_same);
% pair2_same = zeros(pair1_same_l,1);
% for i = 1:pair1_same_l%Ì«ÂýÁË
%     if mod(i,floor(pair1_same_l/10000)) == 0
%         disp([i pair1_same_l]);
%     end;
%     for j = 1:train_num/2
%         if C{2}(pair1_same(i)) == C{2}(pair2(j))
%             pair2_same(i) = pair2(j);
%             break;
%         end;
%     end;
% end;
same_pair_l = floor(train_num / 2 * same_r);
same_in_class = randi(all_class-test_class,floor(same_pair_l * 2),1);
same_in_class_unique = unique(same_in_class);
got_in_same = zeros(train_num,1);
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
diff_num = sum(~got_in_same);
diff_num = floor(diff_num/2) * 2;
diff_sample = find(got_in_same==0);
idx = randperm(diff_num);
diff_sample = diff_sample(idx);
pair1_diff = diff_sample(1:floor(diff_num/2));
pair2_diff = diff_sample(floor(diff_num/2) + 1:end);
err = find(C{2}(pair1_diff) == C{2}(pair2_diff));
% for i = 1 : length(err)
    ridx = randperm(length(err));
    pair1_diff(err) = pair1_diff(err(ridx));
% end;
disp(sum(C{2}(pair1_diff) == C{2}(pair2_diff)));
assert(sum(C{2}(pair1_diff) == C{2}(pair2_diff)) == 0);
pair1 = [pair1_same;pair1_diff];
pair2 = [pair2_same;pair2_diff];
label_sim = [ones(length(pair1_same),1);zeros(length(pair1_diff),1)];
idx = randperm(length(pair1));
pair1 = pair1(idx);
pair2 = pair2(idx);
label_sim = label_sim(idx);
assert(sum((C{2}(pair1) == C{2}(pair2)) ~= label_sim) == 0);
fid1 = fopen('pair_data1.txt','a');
fid2 = fopen('pair_data2.txt','a');
fid3 = fopen('sim_label.txt','a');
for i = 1 : length(pair1)
    fprintf(fid1,'%s %d\r\n',C{1}{pair1(i)},C{2}(pair1(i)));
    fprintf(fid2,'%s %d\r\n',C{1}{pair2(i)},C{2}(pair2(i)));
    fprintf(fid3,'%d\r\n',label_sim(i));
end;
fclose(fid1);
fclose(fid2);
fclose(fid3);