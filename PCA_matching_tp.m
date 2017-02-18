%features
% load 'facefeature_lw6lr1_512.mat'
% features1=features;

%mirror feature
% load 'Mfacefeature_lw6lr1_512.mat'
% features=[features,features1];

tgt_method='facefeature_512'
tgtDir='./LFWTest/';

load pair_lfw
features=double(feature)';
features= bsxfun(@rdivide, features, sqrt(sum(features.^2,2)));

image_path = list;

for i=1:10	
	intra_pairs=pair_lfw.IntraPersonPair{i};
	extra_pairs=pair_lfw.ExtraPersonPair{i};

        train_features=features;
        train_features([intra_pairs(:);extra_pairs(:)],:)=[];
	[coeff,~,latent] = pca(double(train_features)');
        m_feature=mean(double(train_features));
        accuracies = zeros(10,1);

%test the result with the range of PCA components
%	for numDim=64:64:1024
    for numDim=256:256
    %	for numDim=79:104

        test_features1=double(features([intra_pairs(:,1);extra_pairs(:,1)],:));
        test_features2=double(features([intra_pairs(:,2);extra_pairs(:,2)],:));

        test_features1=(test_features1-ones([size(test_features1,1) 1])*m_feature)*coeff(:,1:numDim);
        test_features2=(test_features2-ones([size(test_features2,1) 1])*m_feature)*coeff(:,1:numDim);

        test_features1= bsxfun(@rdivide, test_features1, sqrt(sum(test_features1.^2,2)));
        test_features2= bsxfun(@rdivide, test_features2, sqrt(sum(test_features2.^2,2)));

        scores=diag(test_features1*test_features2');

        test_path1=image_path([intra_pairs(:,1);extra_pairs(:,1)]);
        test_path2=image_path([intra_pairs(:,2);extra_pairs(:,2)]);	
        [tgtDir, '/',tgt_method, '_', num2str(numDim),  '/TR',num2str(i)]
        mkdir([tgtDir, '/',tgt_method, '_', num2str(numDim)]);
        mkdir([tgtDir, '/',tgt_method, '_', num2str(numDim),  '/TR',num2str(i)]);
        
        fid=fopen([tgtDir, '/' ,tgt_method, '_', num2str(numDim), '/TR',num2str(i),'/NCMNCM52evaC.dat'],'w');
        for j=1:length(scores)
            idx=0;
            if ~isempty(strmatch(test_path1{j}(1:end-9),test_path2{j}(1:end-9)))
                idx=1;
            end
            fprintf(fid,'%s\t%s\t1\t%d\t%f\n',test_path1{j}(1:end-4),test_path2{j}(1:end-4),idx,-scores(j));
        end
        fclose(fid);
        accuracy = Match_measure([tgtDir, '/' ,tgt_method, '_', num2str(numDim), '/TR',num2str(i),'/NCMNCM52evaC.dat'])
        accuracies(i,1) = accuracy;
    end
end
mean(accuracy)

