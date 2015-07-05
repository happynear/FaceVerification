if ~exist('model_inited','var')||~model_inited
    caffe.reset_all();
    caffe.set_mode_gpu();
    gpu_id = 0;  % we will use the first gpu in this demo
    caffe.set_device(gpu_id);

%     same_thresh = 1.021622 / 2;
    same_thresh = 0.4;

    % ROIx = 19:82;
    % ROIy = 19:82;
    ROIx = 1:100;
    ROIy = 1:100;

    % ROIx = 1:64;
    % ROIy = 1:64;
    height = length(ROIx);
    width = length(ROIy);
    MatAlignment('init_model','D:\ThirdPartyLibrary\dlib-18.14\shape_predictor_68_face_landmarks.dat');
    meanC = caffe.read_mean('D:\deeplearning\caffe-windows\examples\FaceVerification\mean.proto');
    net = caffe.Net('D:\deeplearning\caffe-windows\examples\FaceVerification\CASIA_demo.prototxt','D:\deeplearning\caffe-windows\examples\FaceVerification\siamese_iter_500000.caffemodel', 'test');
    model_inited = true;
    model_inited64 = false;
end;

% image1 = rgb2gray(imread('D:\deeplearning\caffe-windows\examples\FaceVerification\me1.jpg'));
% image2 = rgb2gray(imread('D:\deeplearning\caffe-windows\examples\FaceVerification\me2.jpg'));
image1 = rgb2gray(imread('H:\»À¡≥\f\0.JPG'));
image2 = rgb2gray(imread('H:\»À¡≥\f\1.JPG'));

J = zeros(height,width,1,2,'single');
face1 = MatAlignment('alignment',image1);

face2 = MatAlignment('alignment',image2);
figure(1);
subplot(1,2,1);
imshow(uint8(face1)');
subplot(1,2,2);
imshow(uint8(face2)');
I = single(face1) - meanC;
I = I(ROIx,ROIy);
J(:,:,1,1) = I/128;
I = single(face2) - meanC;
I = I(ROIx,ROIy);
J(:,:,1,2) = I/128;
H={J};
f = net.forward(H);
f = f{1};
fprintf('distance:%f\n',f);
if f<same_thresh
    fprintf('The two faces are from the same people.\n');
else
    fprintf('The two faces are from different people.\n');
end;