fid = fopen('D:\ThirdPartyLibrary\caffe\list\pair_data1.txt');
C1 = textscan(fid, '%s %d');
fclose(fid);
fid = fopen('D:\ThirdPartyLibrary\caffe\list\pair_data2.txt');
C2 = textscan(fid, '%s %d');
fclose(fid);