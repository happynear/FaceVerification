subset_map = containers.Map;
[a,b,c] = fileparts(image_list{1});
last_folder_all = a;
folder_names = strsplit(a,'\');
last_folder = [folder_names{end-1} '/' folder_names{end}];
subset_map(last_folder) = [1 0];
for i=2:length(image_list)
    [a,b,c] = fileparts(image_list{i});
    folder_names = strsplit(a,'\');
    if strcmp(last_folder_all, a) == 0
        start_end = subset_map(last_folder);
        subset_map(last_folder) = [start_end(1) i-1];
        last_folder = [folder_names{end-1} '/' folder_names{end}];
        subset_map(last_folder) = [i 0];
    end;
    last_folder_all = a;
end;
start_end = subset_map(last_folder);
subset_map(last_folder) = [start_end(1) length(image_list)];