pair_list  = 'E:\datasets\YTF\splits_no_header.txt';
pair_fid = fopen(pair_list,'r');
C = textscan(pair_fid,'%d, %d, %s %s %d');
fclose(pair_fid);
for i=1:length(C{1})
    C{3}{i} = C{3}{i}(1:end-1);
    C{4}{i} = C{4}{i}(1:end-1);
end;

distance_cell = cell(10,1);

for cross = 1:10
    fprintf('%d-th cross validation, collect mean...', cross); 
    train_set1 = C{3}(C{1}~=cross);
    train_set2 = C{4}(C{1}~=cross);
    feature_mean = zeros(512,1);
    feature_count = 0;
    distance_cross = cell(length(train_set1),1);
    feature1_cross = cell(length(train_set1),1);
    feature2_cross = cell(length(train_set1),1);
    for i=1:length(train_set1)
        feature_index = subset_map(train_set1{i});
        feature1 = features(:,feature_index(1):feature_index(2));
        feature_mean = feature_mean + mean(feature1,2);
        feature_index = subset_map(train_set2{i});
        feature2 = features(:,feature_index(1):feature_index(2));
        feature_mean = feature_mean + mean(feature2,2);
        feature_count = feature_count + 2;
        feature1_cross{i} = feature1;
        feature2_cross{i} = feature2;
        if mod(i,450) == 0
            fprintf('%d.', int32(i / 450)); 
        end;
    end;
    fprintf('done.\n');
    fprintf('compute distance...');
    feature_mean = feature_mean / feature_count;
    for i=1:length(train_set1)
%         feature_index = subset_map(train_set1{i});
%         feature1 = features(:,feature_index(1):feature_index(2));
        feature1 = feature1_cross{i};
        feature1 = bsxfun(@minus, feature1, feature_mean);
        feature1 = bsxfun(@rdivide, feature1, sqrt(sum(feature1.^2, 2)));
%         feature_index = subset_map(train_set2{i});
%         feature2 = features(:,feature_index(1):feature_index(2));
        feature2 = feature2_cross{i};
        feature2 = bsxfun(@minus, feature2, feature_mean);
        feature2 = bsxfun(@rdivide, feature2, sqrt(sum(feature2.^2, 2)));
        distance_cross{i} = feature1' * feature2;
        if mod(i,450) == 0
            fprintf('%d.', int32(i / 450)); 
        end;
    end;
    fprintf('done.\n');
    distance_cell{cross} = distance_cross;
end;