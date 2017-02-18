folder = 'E:\datasets\KDEF\aligned_data';
addpath('..');
image_list = get_image_list_in_folder(folder);
list_file = 'E:\datasets\KDEF\list.txt';

emotions = cell(length(image_list),1);
emotion_map = containers.Map;
for i=1:length(image_list)
[a, b, c] = fileparts(image_list{i});
if length(b) >= 6
    emotions{i} = b(5:6);
    emotion_map(b(5:6)) = 1;
end;
end;

emotion_map = containers.Map(emotion_map.keys, [4 1 3 5 0 6 7]);

list_fid = fopen(list_file,'w');
for i=1:length(image_list)
    if ~isempty(emotions{i})
        fprintf(list_fid,'%s %d\r\n', image_list{i}, emotion_map(emotions{i}));
    end;
end;
fclose(list_fid);