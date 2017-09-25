folder = 'F:\datasets\MegaFace\FaceScrub\FaceScrub\';
addpath('..');
% image_list = get_image_list_in_folder(folder);
target_folder = 'D:\datasets\MegaFace\FaceScrub\aligned\';
if exist(target_folder, 'dir')==0
    mkdir(target_folder);
end;

if ~exist('image_list','var')
    actor_file = fopen('F:\datasets\MegaFace\FaceScrub\facescrub_actors_nohead.txt','r');
    actress_file = fopen('F:\datasets\MegaFace\FaceScrub\facescrub_actresses_nohead.txt','r');

    image_list = {};
    boxes = [];
    p=1;
    while ~feof(actor_file)
        str_line = fgetl(actor_file);
        splited = strsplit(str_line,'\t');
        boxes(p,:) = sscanf(splited{end-1},'%d,%d,%d,%d')';
        boxes(p,3) = boxes(p,3) - boxes(p,1);
        boxes(p,4) = boxes(p,4) - boxes(p,2);
        dot_pos = strfind(splited{4},'.');
        dot_pos = dot_pos(end);
        suffix = splited{4}(dot_pos:end);
        if suffix(end) == '&'
            suffix = suffix(1:end-1);
        end;
        if strcmp(suffix, '.gif') ~= 0
            try
                img = imread([folder splited{1} '/' splited{1} '_' splited{3} suffix]);
            catch
                continue;
            end;
            suffix = '.jpg';
            imwrite(img, [folder splited{1} '/' splited{1} '_' splited{3} suffix]);
        end;
        image_list{p} = [folder splited{1} '/' splited{1} '_' splited{3} suffix];
        p=p+1;
    end;
    while ~feof(actress_file)
        str_line = fgetl(actress_file);
        splited = strsplit(str_line,'\t');
        boxes(p,:) = sscanf(splited{end-1},'%d,%d,%d,%d')';
        boxes(p,3) = boxes(p,3) - boxes(p,1);
        boxes(p,4) = boxes(p,4) - boxes(p,2);
        dot_pos = strfind(splited{4},'.');
        dot_pos = dot_pos(end);
        suffix = splited{4}(dot_pos:end);
        if suffix(end) == '&'
            suffix = suffix(1:end-1);
        end;
        if strcmp(suffix, '.gif') ~= 0
            try
                img = imread([folder splited{1} '/' splited{1} '_' splited{3} suffix]);
            catch
                continue;
            end;
            suffix = '.jpg';
            imwrite(img, [folder splited{1} '/' splited{1} '_' splited{3} suffix]);
        end;
        image_list{p} = [folder splited{1} '/' splited{1} '_' splited{3} suffix];
        p=p+1;
    end;
    fclose(actor_file);
    fclose(actress_file);
end;

% list_file = 'F:\datasets\megaface\devkit\templatelists\megaface_features_list.json';
% json_string = fileread(list_file);
% image_list = regexp(json_string(8:end), '"(.*?)"','tokens');
% for i=1:length(image_list)
%     image_list{i} = [folder '/' image_list{i}{1}];
% end;

MTCNN_path = 'E:\Feng\project\MTCNN_face_detection_alignment\code\codes\MTCNNv2';
caffe_model_path=[MTCNN_path , '/model/'];
addpath(genpath(MTCNN_path));

coord5points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041];
imgSize = [112, 96];
            
%caffe.set_mode_cpu();
gpu_id=0;
MatMTCNN('init_model', caffe_model_path, gpu_id);

%three steps's threshold
threshold=[0.6 0.7 0.7];
MatMTCNN('set_threshold', threshold);

minsize = 100;

image_list_len = length(image_list);

for image_id = 1:length(image_list)
    [file_folder, file_name, file_ext] = fileparts(image_list{image_id});
    target_filename = strrep(image_list{image_id},folder, target_folder);
    if exist(target_filename, 'file')
        continue;
    end;
    try
        img = imread(image_list{image_id});
    catch
        continue;
    end;
    if isa(img,'uint16')
        img = uint8(img / 256);
%         imshow(img);
    end;
    if size(img, 3) < 3
       img(:,:,2) = img(:,:,1);
       img(:,:,3) = img(:,:,1);
    end
    if size(img, 1) < 30 || size(img,2) < 30
        continue;
    end;
%     img = img(max(floor(boxes(image_id,2) - (boxes(image_id,4) - boxes(image_id,2)) / 4), 1):...
%         min(floor(boxes(image_id,4) + (boxes(image_id,4) - boxes(image_id,2)) / 4), size(img,1)), ...
%         max(floor(boxes(image_id,1) - (boxes(image_id,3) - boxes(image_id,1)) / 4), 1):...
%         min(floor(boxes(image_id,3) + (boxes(image_id,3) - boxes(image_id,1)) / 4), size(img,2)),:);
%     imshow(img);
    
    assert(strcmp(target_filename, image_list{image_id})==0);
    [file_folder, file_name, file_ext] = fileparts(target_filename);
    if exist(file_folder,'dir')==0
        mkdir(file_folder);
    end;
    
    result = MatMTCNN('detect',img, min([boxes(image_id,3), boxes(image_id,4), size(img,1)/2, size(img,2)/2]));
    default_face = 1;
    if ~isempty(result.bounding_box)
        if size(result.bounding_box,1) > 1
            for i=2:size(result.bounding_box,1)
                if IoU(result.bounding_box(i,:), boxes(image_id,:)) > IoU(result.bounding_box(default_face,:), boxes(image_id,:))
                    default_face = i;
                end;
            end;
        end;
        
        if IoU(result.bounding_box(default_face,:), boxes(image_id,:)) > 0.3
            detected = true;
        else
            detected = false;
        end;
    else
        detected = false;
    end;
    
    if detected
        facial5points = [result.points(default_face,1:2:9);result.points(default_face,2:2:10)];
        Tfm =  cp2tform(facial5points', coord5points', 'similarity');
        cropImg = imtransform(img, Tfm, 'XData', [1 imgSize(2)], 'YData', [1 imgSize(1)], 'Size', imgSize);
        imwrite(cropImg, target_filename);
        fprintf('%d/%d, %s detected=%d, iou=%f\n',image_id,image_list_len,target_filename,size(result.bounding_box,1), IoU(result.bounding_box(default_face,:), boxes(image_id,:)));
    end;
    
%     if detected
%         figure(1);
%         imshow(img);
%         hold on;
%         if ~isempty(result.bounding_box)
%             for j=1:size(result.bounding_box,1)
%                 rectangle('Position',result.bounding_box(j,:),'Edgecolor','g','LineWidth',3); 
%                 plot(result.points(j,1:2:9),result.points(j,2:2:10),'g.','MarkerSize',10)
%             end;
%             rectangle('Position',result.bounding_box(default_face,:),'Edgecolor','r','LineWidth',3); 
%             rectangle('Position',boxes(image_id,:),'Edgecolor','b','LineWidth',3); 
%             if size(result.points,1) >= default_face
%                 plot(result.points(default_face,1:2:9),result.points(default_face,2:2:10),'g.','MarkerSize',10);
%             end;
%         end;
%         hold off;
%         figure(2);
%         imshow(cropImg);
%         pause
%     else
%         figure(1);
%         imshow(img);
%         hold on; 
%         if ~isempty(result.bounding_box)
%             rectangle('Position',boxes(image_id,:),'Edgecolor','b','LineWidth',3);
%             for j=1:size(result.bounding_box,1)
%                 rectangle('Position',result.bounding_box(j,:),'Edgecolor','r','LineWidth',3); 
%                 plot(result.points(j,1:2:9),result.points(j,2:2:10),'g.','MarkerSize',10)
%             end;
%         end;
%         hold off;
%         pause
%     end;

    
end;


function overlap_rate = IoU(bbox1, bbox2)
intersect_bbox = [max(bbox1(1), bbox2(1)) max(bbox1(2), bbox2(2)) min(bbox1(1)+bbox1(3), bbox2(1)+bbox2(3)) min(bbox1(2)+bbox1(4), bbox2(2)+bbox2(4))];
overlap = (intersect_bbox(3) - intersect_bbox(1)) * (intersect_bbox(4) - intersect_bbox(2));
overlap_rate = overlap / (bbox1(3)*bbox1(4) + bbox2(3)*bbox2(4) - overlap);
end
