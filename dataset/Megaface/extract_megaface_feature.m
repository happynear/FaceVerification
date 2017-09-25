caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

batch_size = 50;
feature_dim = 512;
mean_value = 128;
scale = 0.0078125;
ROIx = 1:96;
ROIy = 1:112;
height = length(ROIx);
width = length(ROIy);

feature_suffix = '_hard64nofd';
aligned_image_folder = 'F:\datasets\MegaFace\Megaface\aligned';
feature_folder = ['F:\datasets\MegaFace\Megaface\feature' feature_suffix];
if exist(feature_folder, 'dir')==0
    mkdir(feature_folder);
end;


if ~exist('image_list','var')
    list_file = 'E:\datasets\MegaFace\devkit\templatelists\megaface_features_list.json_1000000_1';
    json_string = fileread(list_file);
    image_list = regexp(json_string(8:end), '"(.*?)"','tokens');
    for i=1:length(image_list)
        image_list{i} = [aligned_image_folder '/' image_list{i}{1}];
    end;
end;

% total_image = length(image_list);
% for image_p=1:total_image
%     target_filename = [strrep(image_list{image_p},aligned_image_folder, feature_folder) feature_suffix '.bin'];
%     if ~exist(target_filename, 'file')
%         break;
%     end;
% end;
% fprintf('start from %d\n',image_p);

image_p = 1;
total_image = length(image_list);
total_iter = ceil((total_image - image_p + 1) / batch_size);

net = caffe.Net('D:\face project\experiment\96_112_l2_distance\face_deploy_64.prototxt','D:\face project\norm_face_model\hard-margin-64-nofd.caffemodel', 'test');%face_model_my

% features = zeros(feature_dim, total_iter * batch_size);
for i=1:total_iter
    if mod(i,100) == 1
        fprintf('%d/%d\n',i, total_iter);
    end;
    J = zeros(height,width,3,batch_size,'single');
    feature_names = cell(batch_size,1);
    for j = 1 : batch_size
        if image_p <= total_image
            I = imread(image_list{image_p});
            if size(I, 3) < 3
               I(:,:,2) = I(:,:,1);
               I(:,:,3) = I(:,:,1);
            end
            I = permute(I,[2 1 3]);
            I = I(:,:,[3 2 1]);
            I = I(ROIx,ROIy,:);
            I = single(I) - mean_value;
            J(:,:,:,j) = I*scale;
            feature_names{j} = [strrep(image_list{image_p},aligned_image_folder, feature_folder) feature_suffix '.bin'];
            image_p = image_p + 1;
        end;
    end;
    f1 = net.forward({J});
    feature = squeeze(f1{1});
%     features(:,(i-1)*batch_size+1:i*batch_size) = feature;
    for j = 1 : batch_size
        if ~isempty(feature_names{j})
            [file_folder, file_name, file_ext] = fileparts(feature_names{j});
            if exist(file_folder,'dir')==0
                mkdir(file_folder);
            end;
            fp = fopen(feature_names{j},'wb');
            fwrite(fp, [feature_dim 1 4 5], 'int32');
            fwrite(fp, feature(:,j), 'float32');
            fclose(fp);
        end;
    end;
end;
% feature1 = features(:,1:250000);
% feature2 = features(:,250001:500000);
% feature3 = features(:,500001:750000);
% feature4 = features(:,750001:1000000);
% save('sphereface_megaface_1.mat','feature1');
% save('sphereface_megaface_2.mat','feature2');
% save('sphereface_megaface_3.mat','feature3');
% save('sphereface_megaface_4.mat','feature4');