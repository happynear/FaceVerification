pair_list  = 'E:\datasets\YTF\splits_no_header.txt';
pair_fid = fopen(pair_list,'r');
C = textscan(pair_fid,'%d, %d, %s %s %d');
fclose(pair_fid);
for i=1:length(C{1})
    C{3}{i} = C{3}{i}(1:end-1);
    C{4}{i} = C{4}{i}(1:end-1);
end;
sampled_map = containers.Map;

sample_size = 15;
train_set1 = C{3};
train_set2 = C{4};
feature_mean = zeros(512,1);
feature_count = 0;
sampled_features = [];
for i=1:length(train_set1)
    feature_index = subset_map(train_set1{i});
    feature1 = features(:,feature_index(1):feature_index(2));
    feature1 = feature1(:, randperm(size(feature1,2), sample_size));
    sampled_features = [sampled_features feature1];
    sampled_map(train_set1{i}) = [size(sampled_features,2) - sample_size + 1 size(sampled_features,2)];
    feature_index = subset_map(train_set2{i});
    feature2 = features(:,feature_index(1):feature_index(2));
    feature2 = feature2(:, randperm(size(feature2,2), sample_size));
    sampled_features = [sampled_features feature2];
    sampled_map(train_set2{i}) = [size(sampled_features,2) - sample_size + 1 size(sampled_features,2)];
    if mod(i,450) == 0
        fprintf('%d.', int32(i / 450)); 
    end;
end;