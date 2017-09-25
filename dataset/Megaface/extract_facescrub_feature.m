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
aligned_image_folder = 'F:\datasets\MegaFace\FaceScrub\aligned\';
feature_folder = ['F:\datasets\MegaFace\FaceScrub\feature' feature_suffix '\'];
if exist(feature_folder, 'dir')==0
    mkdir(feature_folder);
end;


test_list_file = 'E:\datasets\MegaFace\devkit\templatelists\facescrub_uncropped_features_list.json';
json_string = fileread(test_list_file);
json_string = json_string(strfind(json_string,'path')+8:end);
test_list = regexp(json_string(8:end), '"(.*?)"','tokens');
test_list_raw = cell(length(test_list),1);
for i=1:length(test_list)
    test_list_raw{i} = [aligned_image_folder test_list{i}{1}];
    dot_pos = strfind(test_list{i}{1},'.');
    if isempty(dot_pos)||(dot_pos(end) ~= length(test_list{i}{1}) - 3&&dot_pos(end) ~= length(test_list{i}{1}) - 4)
%         disp(test_list{i}{1});
        test_list{i}{1} = [test_list{i}{1} '.jpg'];
    end;
    if ~isempty(dot_pos) && strcmp(test_list{i}{1}(dot_pos(end):end), '.gif')
        test_list{i}{1} = [test_list{i}{1}(1:end-3) 'jpg'];
    end;
    test_list{i} = [aligned_image_folder test_list{i}{1}];
end;

total_image = length(test_list);
total_iter = ceil(total_image / batch_size);

net = caffe.Net('D:\face project\experiment\96_112_l2_distance\face_deploy_64.prototxt','D:\face project\norm_face_model\hard-margin-64-nofd.caffemodel', 'test');%face_model_my

image_p = 1;
for i=1:total_iter
    fprintf('%d/%d\n',i, total_iter);
    J = zeros(height,width,3,batch_size,'single');
    feature_names = cell(batch_size,1);
    for j = 1 : batch_size
        if image_p <= total_image
            I = imread(test_list{image_p});
            I = permute(I,[2 1 3]);
            I = I(:,:,[3 2 1]);
            I = I(ROIx,ROIy,:);
            I = single(I) - mean_value;
            J(:,:,:,j) = I*scale;
            feature_names{j} = [strrep(test_list_raw{image_p},aligned_image_folder, feature_folder) feature_suffix '.bin'];
            image_p = image_p + 1;
        end;
    end;
    f1 = net.forward({J});
    feature = squeeze(f1{1});
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