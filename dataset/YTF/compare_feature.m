pair_list  = 'splits_no_header.txt';
pair_fid = fopen(pair_list,'r');
C = textscan(pair_fid,'%d, %d, %s %s %d');
fclose(pair_fid);
for i=1:length(C{1})
    C{3}{i} = C{3}{i}(1:end-1);
    C{4}{i} = C{4}{i}(1:end-1);
end;

accuracies = zeros(10,1);
distance_dist = cell(10,1);
decisions_mean = zeros(5000,1);
for cross = 1:10
    fprintf('%d-th cross validation, collect mean...', cross); 
    train_set1 = C{3}(C{1}~=cross);
    train_set2 = C{4}(C{1}~=cross);
    feature_mean = zeros(size(feature,1),1);
    feature_count = 0;
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
    distance_cross = zeros(length(C{3}),length(-1:0.02:1) - 1);
    mean_distances = zeros(length(C{3}),1);
    for i=1:length(C{3})
        feature_index = subset_map(C{3}{i});
        feature1 = features(:,feature_index(1):feature_index(2));
%         feature1 = features(:,feature_index(1)-1+randperm(feature_index(2) - feature_index(1)-1, 15));
        feature1 = bsxfun(@minus, feature1, feature_mean);
        feature1 = bsxfun(@rdivide, feature1, sqrt(sum(feature1.^2)));
        feature_index = subset_map(C{4}{i});
        feature2 = features(:,feature_index(1):feature_index(2));
%         feature2 = features(:,feature_index(1)-1+randperm(feature_index(2) - feature_index(1)-1, 15));
        feature2 = bsxfun(@minus, feature2, feature_mean);
        feature2 = bsxfun(@rdivide, feature2, sqrt(sum(feature2.^2)));
       
        distance_matrix = feature1' * feature2;
        histo = histcounts(distance_matrix(:),-1:0.02:1,'Normalization','probability');
        distance_cross(i,:) = histo;
        mean_distances(i) = mean(distance_matrix(:));
        if mod(i,500) == 0
            fprintf('%d.', int32(i / 500)); 
        end;
    end;
    distance_dist{cross} = distance_cross;
    fprintf('done.\n');
    cmd = [' -t 0 -h 0'];
    model = svmtrain(double(C{5}(C{1}~=cross)),mean_distances(C{1}~=cross),cmd);
    [class, accuracy, deci] = svmpredict(double(C{5}(C{1}==cross)),mean_distances(C{1}==cross),model);
%     cmd = [' -t 2 -h 0'];
%     model = svmtrain(double(C{5}(C{1}~=cross)),distance_cross(C{1}~=cross,:),cmd);
%     [class, accuracy, deci] = svmpredict(double(C{5}(C{1}==cross)),distance_cross(C{1}==cross,:),model);
    fprintf('%d th-fold accuracy:%.4f\n', cross, accuracy(1));
    accuracies(cross) = accuracy(1);
    decisions_mean((cross-1)*500+1:cross*500) = deci;
end;
fprintf('accuracy by 10-fold evalution:%.4f\n', mean(accuracies));
roc_curve(decisions_mean, C{5}*2-1);