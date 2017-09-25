%% Align megaface with the help of the provided 3 points.
%% I got 739,181 out of 1M aligned faces using this code.
%% The left faces need to use align_megaface_failures.m to align
folder = 'E:\datasets\megaface\megafacedata\FlickrFinal2';
if ~exist('image_list','var')
    list_file = 'E:\datasets\megaface\devkit\templatelists\megaface_features_list.json_10000_1';
    json_string = fileread(list_file);
    image_list = regexp(json_string(8:end), '"(.*?)"','tokens');
    for i=1:length(image_list)
        image_list{i} = [folder '/' image_list{i}{1}];
    end;
end;

target_folder = 'F:\datasets\MegaFace\Megaface\aligned';
if exist(target_folder, 'dir')==0
    mkdir(target_folder);
end;

pdollar_toolbox_path='D:/face project/pdollar-toolbox';
addpath(genpath(pdollar_toolbox_path));

MTCNN_path = 'D:\face project\MTCNN_face_detection_alignment\code\codes\MTCNNv2';
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
landmark_list = cell(image_list_len,1);

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
    
%     result1 = MatMTCNN('detect', img, min(imgSize(1), imgSize(2)));
%     numbox1=size(result1.bounding_box,1);
%     original_img = img;
    
    assert(strcmp(target_filename, image_list{image_id})==0);
    [file_folder, file_name, file_ext] = fileparts(target_filename);
    if exist(file_folder,'dir')==0
        mkdir(file_folder);
    end;
    
    json = parse_json(fileread([image_list{image_id} '.json']));
    
    if isfield(json{1}, 'landmarks')
        landmarks = json{1}.landmarks;
        landmarks_fields = fieldnames(json{1}.landmarks);
        if length(landmarks_fields) == 3
            try
            facial3points = [json{1}.landmarks.n1.x json{1}.landmarks.n0.x json{1}.landmarks.n2.x;
                json{1}.landmarks.n1.y json{1}.landmarks.n0.y json{1}.landmarks.n2.y];
            catch
                continue;
            end;
            Tfm =  cp2tform(facial3points', coord5points(:,1:3)'*1.5+repmat(imgSize*0.25,[3 1]), 'similarity');
            img = imtransform(img, Tfm, 'XData', [1 imgSize(1)*2],...
                                          'YData', [1 imgSize(1)*2], 'Size', [imgSize(1)*2 imgSize(1)*2]);
            result = MatMTCNN('detect', img, imgSize(1));
            default_face = 1;
            if ~isempty(result.bounding_box)
                facial5points = [result.points(default_face,1:2:9);result.points(default_face,2:2:10)];
                Tfm =  cp2tform(facial5points', coord5points', 'similarity');
                cropImg = imtransform(img, Tfm, 'XData', [1 imgSize(2)],...
                                              'YData', [1 imgSize(1)], 'Size', imgSize);
                fprintf('%d/%d, %s [%d,%d]\n',image_id,image_list_len,target_filename,int32(result.bounding_box(default_face,3)),int32(result.bounding_box(default_face,4)));
            else
                continue;
            end;   
        end;
    end;
    
    if exist('cropImg','var')
        imwrite(cropImg, target_filename);
    end;
%     	show detection result
%     if numbox1 > 0 && ~exist('cropImg','var')
%         figure(3);
%         imshow(original_img);
%         hold on; 
%         if ~isempty(result1.bounding_box)
%             for j=1:numbox1
%                 rectangle('Position',result1.bounding_box(j,:),'Edgecolor','r','LineWidth',3); % Note that the predicted bbox could still not close to labeled bbox
%                 plot(result1.points(j,1:2:9),result1.points(j,2:2:10),'g.','MarkerSize',10);
%             end;
%         end;
%         hold off;
%         pause
%     end;
%     if numbox1 ==0 && exist('cropImg','var')
%         numbox=size(result.bounding_box,1);
%         figure(1);
%         imshow(img);
%         hold on; 
%         if ~isempty(result.bounding_box)
%             for j=1:numbox
%                 rectangle('Position',result.bounding_box(default_face,:),'Edgecolor','r','LineWidth',3); % Note that the predicted bbox could still not close to labeled bbox
%                 plot(result.points(default_face,1:2:9),result.points(default_face,2:2:10),'g.','MarkerSize',10);
%             end;
%         end;
%         hold off;
%         figure(2);
%         imshow(cropImg);
%         figure(3);
%         imshow(original_img);
%         hold on; 
%         if ~isempty(result1.bounding_box)
%             for j=1:numbox1
%                 rectangle('Position',result1.bounding_box(j,:),'Edgecolor','r','LineWidth',3); % Note that the predicted bbox could still not close to labeled bbox
%                 plot(result1.points(j,1:2:9),result1.points(j,2:2:10),'g.','MarkerSize',10);
%             end;
%         end;
%         hold off;
%         pause
%         
%     end;
end;
