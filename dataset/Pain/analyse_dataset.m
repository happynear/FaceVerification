list = 'list_val.txt';
list_fid = fopen(list,'r');
C = textscan(list_fid,'%s %d');
fclose(list_fid);
label = double(C{2});
hist(label,15);
mean(label.^2)
mean((label - mean(label)).^2)