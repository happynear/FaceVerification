count = 10;
fid = fopen('list.txt');
C = textscan(fid, '%s %d');
all_class = 10575;
image_num_in_class = zeros(all_class,1);
image_in_class = cell(all_class,1);
for i = 1:all_class
    image_in_class{i} = find(C{2} == i -1);
    image_num_in_class(i) = length(image_in_class{i});
end;
fclose(fid);
for i=1:100
    try
        getTrainList(C,image_in_class,image_num_in_class);
        count=count-1;
    catch
        
    end;
    disp(count);
    if count==0
        break;
    end;
end;