target_folder = 'aligned_images';

pdollar_toolbox_path='D:/face project/pdollar-toolbox';
addpath(genpath(pdollar_toolbox_path));

MTCNN_path = 'D:\face project\MTCNN_face_detection_alignment\code\codes\MTCNNv2';
caffe_model_path=[MTCNN_path , '/model'];
addpath(genpath(MTCNN_path));

coord5points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041];
imgSize = [112, 96];
            
%caffe.set_mode_cpu();
gpu_id=0;
caffe.set_mode_gpu();	
caffe.set_device(gpu_id);
caffe.reset_all();

%three steps's threshold
threshold=[0.6 0.7 0.7]

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

identity_list = dir('./Images');
identity_list = identity_list(3:end);

for i=1:length(identity_list)
    subset_list = dir(fullfile('./Images',identity_list(i).name));
    subset_list = subset_list(3:end);
    for j=1:length(subset_list)
        image_list = dir(fullfile('./Images',identity_list(i).name, subset_list(j).name));
        image_list = image_list(3:end);
        disp([fullfile('./Images',identity_list(i).name, subset_list(j).name) ' ' num2str(length(image_list)) ' files']);
        for k=1:length(image_list)
            filename = fullfile('./Images',identity_list(i).name, subset_list(j).name, image_list(k).name);
            [~, ~, c] = fileparts(filename);
            if strcmp(c,'.png')==0
                continue;
            end;
            img = imread(filename);
            if size(img,1) == 112 && size(img,2) == 96
                continue;
            end;
            [boundingboxes points]=detect_face(img,minsize,PNet,RNet,ONet,LNet,threshold,false,factor);
            if isempty(boundingboxes)
                continue;
            end;
            default_face = 1;
            if size(boundingboxes,1) > 1
                for bb=2:size(boundingboxes,1)
                    if abs((boundingboxes(bb,1) + boundingboxes(bb,3))/2 - size(img,2) / 2) + abs((boundingboxes(bb,2) + boundingboxes(bb,4))/2 - size(img,1) / 2) < ...
                            abs((boundingboxes(default_face,1) + boundingboxes(default_face,3))/2 - size(img,2) / 2) + abs((boundingboxes(default_face,2) + boundingboxes(default_face,4))/2 - size(img,1) / 2)
                        default_face = bb;
                    end;
                end;
            end;
            facial5points = double(reshape(points(:,default_face),[5 2])');
            Tfm =  cp2tform(facial5points', coord5points', 'similarity');
            cropImg = imtransform(img, Tfm, 'XData', [1 imgSize(2)],...
                                          'YData', [1 imgSize(1)], 'Size', imgSize);
            imwrite(cropImg, filename);
            %show detection result
        % 	numbox=size(boundingboxes,1);
        %     figure(1);
        % 	imshow(img)
        % 	hold on; 
        % 	for j=1:numbox
        % 		plot(points(1:5,j),points(6:10,j),'g.','MarkerSize',10);
        % 		r=rectangle('Position',[boundingboxes(j,1:2) boundingboxes(j,3:4)-boundingboxes(j,1:2)],'Edgecolor','g','LineWidth',3);
        %     end
        %     hold off;
        %     figure(2);
        %     imshow(cropImg);
        % 	pause

        end;
    end;
end;

