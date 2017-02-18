list_filename = 'MS-Celeb-1M_clean_list.txt';
source_folder = 'data';
target_folder = 'C:\datasets\MS-Celeb-1M-aligned\';

list_fid = fopen(list_filename, 'r');
C = textscan(list_fid,'%s %d');
fclose(list_fid);

if exist(target_folder,'dir')==0
    mkdir(target_folder);
end;

current_class = -1;
total_class = C{2}(end);
total_lines = length(C{1});
for i=1:total_lines
    if C{2}(i) ~= current_class
        current_class = C{2}(i);
        all_files = C{1}(C{2} == current_class);
        folder_map = containers.Map;
        for j = 1:length(all_files)
            folder_filename = strsplit(all_files{j},'/');
            if isKey(folder_map, folder_filename{1})
                folder_map(folder_filename{1}) = folder_map(folder_filename{1}) + 1;
            else
                folder_map(folder_filename{1}) = 1;
            end;
        end;
        map_keys = folder_map.keys;
        map_values = folder_map.values;
        [~,p] = max(map_values{1});
        assert(p==1);
        folder_name = map_keys{p};
        current_folder = fullfile(target_folder,folder_name);
        if exist(current_folder,'dir')==0
            mkdir(current_folder);
        end;
    end;
    if exist(fullfile(source_folder, C{1}{i}),'file') > 0
        folder_filename_target = strsplit(C{1}{i},'/');
        target_file = fullfile(current_folder, folder_filename_target{2});
        copyfile(fullfile(source_folder, C{1}{i}), target_file);
        disp([num2str(i) '/' num2str(total_lines) ' ' num2str(current_class) '/' num2str(total_class) ' ' C{1}{i}]);
    end;
end;