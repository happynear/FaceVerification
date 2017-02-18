identity_list = dir('./Images');
identity_list = identity_list(3:end);
list = 'list.txt';
list_fid = fopen(list,'w');

for i=1:length(identity_list)
    subset_list = dir(fullfile('./Images',identity_list(i).name));
    subset_list = subset_list(3:end);
    for j=1:length(subset_list)
        image_list = dir(fullfile('./Images',identity_list(i).name, subset_list(j).name));
        image_list = image_list(3:end);
        disp([fullfile('./Images',identity_list(i).name, subset_list(j).name) ' ' num2str(length(image_list)) ' files']);
        for k=1:length(image_list)
            filename = fullfile(pwd, 'Images',identity_list(i).name, subset_list(j).name, image_list(k).name);     
            [a, b, c] = fileparts(filename);
            if strcmp(c,'.png')==0
                continue;
            end;
%             image = imread(filename);
%             assert(size(image,1)==112 && size(image,2)==96);
            label_file = fullfile('./Frame_Labels/PSPI',identity_list(i).name, subset_list(j).name, [b '_facs.txt']);
            label = textread(label_file);
            fprintf(list_fid,'%s %d\r\n', fullfile(identity_list(i).name, subset_list(j).name, image_list(k).name), uint8(label));
        end;
    end;
end;
fclose(list_fid);