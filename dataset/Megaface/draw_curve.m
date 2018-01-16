result_folder = 'E:\datasets\MegaFace\results\';
feature_suffix = '_cos-m35';
facescrub_cmc_file = ['cmc_facescrub_megaface' feature_suffix '_1000000_1.json'];
facescrub_matches_file = ['matches_facescrub_megaface' feature_suffix '_1000000_1.json'];
% facescrub_cmc_file = 'E:\datasets\MegaFace\results\Challenge1External\facenet\cmc_megaface_facescrub_1000000_1.json';

facescrub_cmc_json = loadjson(fileread([result_folder facescrub_cmc_file]));
% facescrub_cmc_json = loadjson(fileread(facescrub_cmc_file));

figure(1);
semilogx(facescrub_cmc_json.cmc{1}+1,facescrub_cmc_json.cmc{2}*100,'LineWidth',2);
title(['Identification @ 1e6 distractors = ' num2str(facescrub_cmc_json.cmc{2}(1))]);
xlabel('Rank');
ylabel('Identification Rate %');
grid on;
box on;
hold on;

figure(2);
semilogx(facescrub_cmc_json.roc{1},facescrub_cmc_json.roc{2},'LineWidth',2);
title(['Verification @ 1e-6 = ' num2str(interp1(facescrub_cmc_json.roc{1}, facescrub_cmc_json.roc{2}, 1e-6))]);
xlim([1e-6 1]);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
grid on;
box on;
hold on;