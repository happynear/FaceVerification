base64 = org.apache.commons.codec.binary.Base64;
tsv_file = 'MsCelebV1-Faces-Cropped.Samples.tsv';

root_folder = 'data';
mkdir(root_folder);

tsv_fid = fopen(tsv_file, 'r');
total_lines = 3481187;
image_id = 1;
while ~feof(tsv_fid)
    line = fgetl(tsv_fid);%MID,EntityNameString,ImageURL,FaceID,FaceRectangle_Base64Encoded,FaceData_Base64Encoded
    C = strsplit(line,'\t');
    folder = fullfile(root_folder,C{1});
    if exist(folder,'dir')==0
        mkdir(folder);
    end;
    filename = fullfile(folder,[C{2} '-' C{5} '.jpg']);
    raw  = base64decode(C{7});
%     img_fid = fopen(filename, 'wb');
%     fwrite(img_fid, raw, 'uint8');
%     fclose(img_fid);
    jImg = javax.imageio.ImageIO.read(java.io.ByteArrayInputStream(raw));
    h = jImg.getHeight;
    w = jImg.getWidth;
    p = typecast(jImg.getData.getDataStorage, 'uint8');
    img = permute(reshape(p, [3 w h]), [3 2 1]);
    img = img(:,:,[3 2 1]);
    imshow(img);
    disp([num2str(image_id) '/' num2str(total_lines) ' ' filename]);
    image_id = image_id + 1;
end;
fclose(tsv_fid);