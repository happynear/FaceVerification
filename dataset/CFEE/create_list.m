folder = 'E:\datasets\CFEE\aligned_data';
addpath('..');
image_list = get_image_list_in_folder(folder);
list_file = 'E:\datasets\CFEE\\list.txt';

list_fid = fopen(list_file,'w');
for i=1:length(image_list)
    [a,b,c] = fileparts(image_list{i});
    fprintf(list_fid,'%s %d\r\n', image_list{i}, str2double(b(1:2))-1);
end;
fclose(list_fid);