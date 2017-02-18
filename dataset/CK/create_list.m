folder = 'E:\datasets\cohn-kanade-images\keep-align-kaipeng';
addpath('..');
image_list = get_image_list_in_folder(folder);
list_file = 'E:\datasets\cohn-kanade-images\list.txt';

list_fid = fopen(list_file,'w');
for i=1:length(image_list)
    [a,b,c]=fileparts(image_list{i});
    fprintf(list_fid,'%s %c\r\n', image_list{i}, b(end));
end;
fclose(list_fid);