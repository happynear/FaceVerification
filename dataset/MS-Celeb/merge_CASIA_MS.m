target_ms_folder = 'C:\datasets\MS-Celeb-1M-aligned';
casia_folder = 'C:\datasets\CASIA-maxpy-clean-aligned-96';
same_person_list = 'E:\datasets\Ms-Celeb-1M\webface_id_name_mid_lite.txt';

same_list_fid = fopen(same_person_list, 'r');
C = textscan(same_list_fid, '%s %s %s');
fclose(same_list_fid);
copied_person = containers.Map;
temp_folder = 'C:\datasets\casia-temp';
if exist(temp_folder,'dir') == 0
    mkdir(temp_folder);
end;

for i=1:length(C{1})
    subset_list = dir(fullfile(casia_folder,C{1}{i}));
    subset_list = subset_list(3:end);
    if exist(fullfile(temp_folder, C{3}{i}),'dir') == 0
        mkdir(fullfile(temp_folder, C{3}{i}));
        fprintf('created %s\r\n',fullfile(temp_folder, C{3}{i}));
    end;
    fprintf('merging %s\r\n', C{2}{i});
    for j = 1:length(subset_list)
        copyfile(fullfile(casia_folder, C{1}{i}, subset_list(j).name), fullfile(temp_folder, C{3}{i}, subset_list(j).name));
    end;
    
    copied_person(C{1}{i}) = 1;
end;

casia_list = fopen(fullfile(casia_folder, 'list.txt'), 'r');
C2 = textscan(casia_list, '%s %d');
fclose(casia_list);


for i=1:length(C2{1})
    folder_file = strsplit(C2{1}{i},'/');
    if mod(i,100)==0
        disp([i, length(C2{1})]);
    end;
    if ~isKey(copied_person,folder_file(1))
        if exist(fullfile(temp_folder, folder_file{1}),'dir')==0
            mkdir(fullfile(temp_folder, folder_file{1}));
        end;
        copyfile(fullfile(casia_folder, C2{1}{i}), fullfile(temp_folder, C2{1}{i}));
    end;
end;