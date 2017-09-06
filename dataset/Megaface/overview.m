folder = 'F:\datasets\megaface\megafacedata\FlickrFinal2';
if ~exist('image_list','var')
    list_file = 'F:\datasets\megaface\devkit\templatelists\megaface_features_list.json';
    json_string = fileread(list_file);
    image_list = regexp(json_string(8:end), '"(.*?)"','tokens');
    for i=1:length(image_list)
        image_list{i} = [folder '/' image_list{i}{1}];
    end;
end;

target_folder = 'D:\datasets\MegaFace\megafacedata\aligned';
if exist(target_folder, 'dir')==0
    mkdir(target_folder);
end;

MTCNN_path = 'E:\Feng\project\MTCNN_face_detection_alignment\code\codes\MTCNNv2';
caffe_model_path=[MTCNN_path , '/model/'];
addpath(genpath(MTCNN_path));
gpu_id = 1;
MatMTCNN('init_model', caffe_model_path, gpu_id);

image_list_len = length(image_list);
view_order = randperm(image_list_len);

coord5points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041];
imgSize = [112, 96];

for image_id = view_order
    [file_folder, file_name, file_ext] = fileparts(image_list{image_id});
    target_filename = strrep(image_list{image_id},folder, target_folder);
    if ~exist([image_list{image_id}, '.json'],'file')
        continue;
    end;
    try
        img = imread(image_list{image_id});
    catch
        continue;
    end;
    if size(img, 3) < 3
       img(:,:,2) = img(:,:,1);
       img(:,:,3) = img(:,:,1);
    end
    json = parse_json(fileread([image_list{image_id} '.json']));

    all_point_names = fieldnames(json{1}.landmarks);
    if length(all_point_names)==3
        continue;
    end;
    if ~isfield(json{1}, 'bounding_box')
        break;
    else
        continue;
    end;
    figure(1);
    imshow(img);
    hold on;
    for p = 1:length(all_point_names)
        point_p = json{1}.landmarks.(all_point_names{p});
        plot(point_p.x,point_p.y,'g.','MarkerSize',10);
    end;
    if isfield(json{1}, 'bounding_box')
        rectangle('Position',[json{1}.bounding_box.x json{1}.bounding_box.y json{1}.bounding_box.width json{1}.bounding_box.height],'Edgecolor','r','LineWidth',3);
    end;
    hold off;
    
    if isfield(json{1}, 'bounding_box')
        result = MatMTCNN('force_detect',img, double([json{1}.bounding_box.x json{1}.bounding_box.y json{1}.bounding_box.width json{1}.bounding_box.height]));
        if ~isempty(result.bounding_box)
            figure(2);
            imshow(img);
            hold on;
            rectangle('Position',result.bounding_box,'Edgecolor','r','LineWidth',3);
            for i=1:5
                plot(result.points(1,2*i-1),result.points(1,2*i),'g.','MarkerSize',10);
            end;
            hold off;
        end;
    end;
    
    result = MatMTCNN('detect',img, size(img,1)/4);
    default_face = 1;
    if ~isempty(result.bounding_box)
        figure(3);
        imshow(img);
        hold on;
        if size(result.bounding_box,1) > 1
            if isfield(json{1}, 'bounding_box') % select bbox via IoU
                coarse_box = [json{1}.bounding_box.x json{1}.bounding_box.y json{1}.bounding_box.width json{1}.bounding_box.height];
                for i=2:size(result.bounding_box,1)
                    if IoU(result.bounding_box(i,:), coarse_box) > IoU(result.bounding_box(i,:), coarse_box)
                        default_face = i;
                    end;
                end;
            else % select bbox most close to center
                for i=2:size(boundingboxes,1)
                    if Distance2Center(result.bounding_box(i,:), [size(img,1) size(img,2)]) < Distance2Center(result.bounding_box(default_face,:), [size(img,1) size(img,2)])
                        default_face = i;
                    end;
                end;
            end;
        end;
        rectangle('Position',result.bounding_box(default_face,:),'Edgecolor','r','LineWidth',3); % Note that the predicted bbox could still not close to labeled bbox
        for i=1:5
            plot(result.points(default_face,2*i-1),result.points(default_face,2*i),'g.','MarkerSize',10);
        end;
        hold off;
    end;
    if ~isempty(result.points)
        facial5points = [result.points(default_face,1:2:9);result.points(default_face,2:2:10)];
        Tfm =  cp2tform(facial5points', coord5points', 'similarity');
        cropImg = imtransform(img, Tfm, 'XData', [1 imgSize(2)], 'YData', [1 imgSize(1)], 'Size', imgSize);
        figure(4);
        imshow(cropImg);
    end;
    pause;
end;

function overlap_rate = IoU(bbox1, bbox2)
intersect_bbox = [max(bbox1(1), bbox2(1)) max(bbox1(2), bbox2(2)) min(bbox1(1)+bbox1(3), bbox2(1)+bbox2(3)) min(bbox1(2)+bbox1(4), bbox2(2)+bbox2(4))];
overlap = (intersect_bbox(3) - intersect_bbox(1)) * (intersect_bbox(4) - intersect_bbox(2));
overlap_rate = overlap / (bbox1(3)*bbox1(4) + bbox2(3)*bbox2(4) - overlap);
end

function center_dis = Distance2Center(bbox, imgSize)
bbox_center = [bbox(1) + bbox(3)/2 bbox(2) + bbox(4)/2];
img_center = imgSize / 2;
center_dis = norm(bbox_center - img_center);
end