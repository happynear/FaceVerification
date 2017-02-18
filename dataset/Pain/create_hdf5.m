identity_list = dir('./Images');
identity_list = identity_list(3:end);

labels = [];
h5_id = 0;
hdf5_list = 'hdf5_list.txt';
hdf5_list_fid = fopen(hdf5_list,'w');
% h5create(['hdf5_subset' num2str(h5_id) '.h5'], '/data', [112 96 3 Inf],'Datatype','single');
% h5create(['hdf5_subset' num2str(h5_id) '.h5'], '/label', [1 Inf],'Datatype','single');
fprintf(hdf5_list_fid, 'C:/datasets/Pain/hdf5/%s\n', ['hdf5_subset' num2str(h5_id) '.h5']);
stored_files = 0;
created_flag=false;
chunksz = 1000;
images = zeros(112, 96, 3, 1000,'single');
labels = zeros(1,1000,'single');

for i=1:length(identity_list)
    subset_list = dir(fullfile('./Images',identity_list(i).name));
    subset_list = subset_list(3:end);
    for j=1:length(subset_list)
        image_list = dir(fullfile('./Images',identity_list(i).name, subset_list(j).name));
        image_list = image_list(3:end);
        disp([fullfile('./Images',identity_list(i).name, subset_list(j).name) ' ' num2str(length(image_list)) ' files']);
        for k=1:length(image_list)
            filename = fullfile('./Images',identity_list(i).name, subset_list(j).name, image_list(k).name);
            [a, b, c] = fileparts(filename);
            if strcmp(c,'.png')==0
                continue;
            end;
            label_file = fullfile('./Frame_Labels/PSPI',identity_list(i).name, subset_list(j).name, [b '_facs.txt']);
            label = textread(label_file);
            image = imread(filename);
            if size(image,1) ~= 112 || size(image,2) ~= 96
                continue;
            end;
            stored_files = stored_files + 1;
            images(:,:,:,stored_files) = (single(image) - 128)*0.04;
            labels(stored_files) = single(label);
            if stored_files == 1000
                curr_dat_sz=store2hdf5(['hdf5/hdf5_subset' num2str(h5_id) '.h5'], images, labels, ~created_flag);
                created_flag=true;% flag set so that file is created only once
                stored_files = 0;
                h5_id = h5_id + 1;
                fprintf(hdf5_list_fid, 'C:/datasets/Pain/hdf5/%s\n', ['hdf5_subset' num2str(h5_id) '.h5']);
                created_flag=false;
            end;
        end;
    end;
end;
curr_dat_sz=store2hdf5(['hdf5/hdf5_subset' num2str(h5_id) '.h5'], images(:,:,:,1:stored_files), labels(1:stored_files), ~created_flag);
fclose(hdf5_list_fid);