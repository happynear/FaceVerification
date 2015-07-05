check_label = (class_casia == same_label) & (class ~= class_casia);
check_x = find(check_label==1);
allPairs = [same_pair;diff_pair];

num_window = 10;
border = 10;
for i=1:length(check_x) / num_window
    figure(1);
    background = uint8(ones(num_window*(100+border)+border, 2*(100+border)+border))*255;
    for j = 1 : num_window
        idx = check_x((i-1)*num_window+j);
        I1 = imread(allPairs{idx,1});
        I2 = imread(allPairs{idx,2});
%         subplot(num_window,2,(j-1)*2 + 1);
%         imshow(I1);
%         subplot(num_window,2,(j-1)*2 + 2);
%         imshow(I2);
        background((j-1)*(100+border)+border+1:j*(100+border),border+1:100+border) = I1;
        background((j-1)*(100+border)+border+1:j*(100+border),border*2 + 100+1:200+2*border) = I2;
    end;
    imshow(background);
    input('')
end;

fid = fopen('G:\lfw\list.txt','w');
for i = 1:length(check_x)
    idx = check_x(i);
    fprintf(fid,'%s %d\r\n',allPairs{idx,1},lfw_label(idx,1));
    fprintf(fid,'%s %d\r\n',allPairs{idx,2},lfw_label(idx,2));
end;
fclose(fid);