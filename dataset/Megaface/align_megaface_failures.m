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

pdollar_toolbox_path='D:/face project/pdollar-toolbox';
addpath(genpath(pdollar_toolbox_path));

MTCNN_path = 'E:\Feng\project\MTCNN_face_detection_alignment\code\codes\MTCNNv2';
caffe_model_path=[MTCNN_path , '/model/'];
addpath(genpath(MTCNN_path));
gpu_id = 0;
MatMTCNN('init_model', caffe_model_path, gpu_id);

coord5points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041];
imgSize = [112, 96];

%three steps's threshold
threshold=[0.6 0.7 0.7]
MatMTCNN('set_threshold', threshold);
minsize = 100;

%scale factor
factor=0.709;

%load caffe models
prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
model_dir = strcat(caffe_model_path,'/det1.caffemodel');
PNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
model_dir = strcat(caffe_model_path,'/det2.caffemodel');
RNet=caffe.Net(prototxt_dir,model_dir,'test');	
prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
model_dir = strcat(caffe_model_path,'/det3.caffemodel');
ONet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir =  strcat(caffe_model_path,'/det4.prototxt');
model_dir =  strcat(caffe_model_path,'/det4.caffemodel');
LNet=caffe.Net(prototxt_dir,model_dir,'test');
faces=cell(0);	

image_list_len = length(image_list);

for image_id = 1:image_list_len
    clear cropImg;
    [file_folder, file_name, file_ext] = fileparts(image_list{image_id});
    target_filename = strrep(image_list{image_id},folder, target_folder);
    if ~exist([image_list{image_id}, '.json'],'file')
        continue;
    end;
    if exist(target_filename, 'file')
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
    
    assert(strcmp(target_filename, image_list{image_id})==0);
    [file_folder, file_name, file_ext] = fileparts(target_filename);
    if exist(file_folder,'dir')==0
        mkdir(file_folder);
    end;
    
    json = parse_json(fileread([image_list{image_id} '.json']));
    coarse_box = [json{1}.bounding_box.x json{1}.bounding_box.y json{1}.bounding_box.width json{1}.bounding_box.height];
    
    result = MatMTCNN('detect',img, size(img,1)/4);
    default_face = 1;
    if ~isempty(result.bounding_box)
        if size(result.bounding_box,1) > 1
            for i=2:size(result.bounding_box,1)
                if IoU(result.bounding_box(i,:), coarse_box) > IoU(result.bounding_box(default_face,:), coarse_box)
                    default_face = i;
                end;
            end;
        end;
        
        if IoU(result.bounding_box(default_face,:), coarse_box) > 0.3
            force_detect = false;
        else
            force_detect = true;
        end;
    else
        force_detect = true;
    end;
    
    if force_detect
        result = MatMTCNN('force_detect',img,coarse_box);
        if result.score(1) < 0.3
            result.bounding_box = [];
            result.score = [];
            result.points = [];
        end;
    end;
    
    if ~isempty(result.points)
        facial5points = [result.points(default_face,1:2:9);result.points(default_face,2:2:10)];
        Tfm =  cp2tform(facial5points', coord5points', 'similarity');
        cropImg = imtransform(img, Tfm, 'XData', [1 imgSize(2)], 'YData', [1 imgSize(1)], 'Size', imgSize);
        fprintf('%d/%d, %s force_detect=%d, detected=%d, iou=%f\n',image_id,image_list_len,target_filename,int32(force_detect),size(result.bounding_box,1), IoU(result.bounding_box(default_face,:), coarse_box));
    else
        roi = coarse_box;
        roi(3) = coarse_box(4) / imgSize(1) * imgSize(2);
        roi(1) = coarse_box(1) + (coarse_box(3) - roi(3)) / 2;
        result.bounding_box = floor(roi);
        default_face = 1;
        cropImg = imcrop(img,roi);
        cropImg = imresize(cropImg,imgSize);
        fprintf('%d/%d, %s force_detect=%d, detected=%d\n',image_id,image_list_len,target_filename,int32(force_detect),size(result.bounding_box,1));
    end;
    
    
    
    if exist('cropImg','var')
        imwrite(cropImg, target_filename);
    end;
% 
%     if exist('cropImg','var')
%         numbox=size(result.bounding_box,1);
%         figure(1);
%         imshow(img);
%         hold on; 
%         if ~isempty(result.bounding_box)
%             for j=1:numbox
%                 rectangle('Position',result.bounding_box(default_face,:),'Edgecolor','r','LineWidth',3); % Note that the predicted bbox could still not close to labeled bbox
%                 rectangle('Position',coarse_box,'Edgecolor','b','LineWidth',3); % Note that the predicted bbox could still not close to labeled bbox
%                 if size(result.points,1) >= default_face
%                     plot(result.points(default_face,1:2:9),result.points(default_face,2:2:10),'g.','MarkerSize',10);
%                 end;
%             end;
%         end;
%         hold off;
%         figure(2);
%         imshow(cropImg);
%         pause
%     end;
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