#include <io.h>
#include <direct.h>
#include <string>
#include <iomanip>
#include <Windows.h>
#include <AtlBase.h>
#include <opencv2/opencv.hpp>

#include "stdafx.h"
//https://github.com/ShiqiYu/libfacedetection
#include "facedetect-dll.h"
//#pragma comment(lib,"libfacedetect.lib")


using namespace cv;
using namespace std;

void getFiles( string path, vector<pair<string,int> >& files ) {
    //文件句柄  
    long   hFile   =   0;  
    //文件信息  
    struct _finddata_t fileinfo;  

    string p;

	static int auto_label = 1;
	int label=0;
	p.clear();
	FILE *fp = fopen(p.assign(path).append("/label.txt").c_str(),"r");
	if(fp==NULL)
	{
		label = auto_label++;
		cout<<path<<" label:"<<label<<endl;
	}
	else
	{
		fscanf(fp,"%d",&label);
		if(auto_label<label) auto_label = label + 1;
		fclose(fp);
	}


    if   ((hFile   =   _findfirst(p.assign(path).append("/*").c_str(),&fileinfo))   !=   -1)  {  

        do  {  
            //如果是目录,迭代之
            //如果不是,加入列表
            if   ((fileinfo.attrib   &   _A_SUBDIR)) {  
                if   (strcmp(fileinfo.name,".")   !=   0   &&   strcmp(fileinfo.name,"..")   !=   0)  
                    getFiles(   p.assign(path).append("/").append(fileinfo.name), files   );  
            }  else  {  
                if(strstr(fileinfo.name,"png")!=NULL||strstr(fileinfo.name,"bmp")!=NULL||strstr(fileinfo.name,"jpg")!=NULL||strstr(fileinfo.name,"tif")!=NULL)
                    files.push_back( make_pair(p.assign(path).append("\\").append(fileinfo.name),label) );
            }  
        }   while   (_findnext(   hFile,   &fileinfo   )   ==   0);  

        _findclose(hFile);  
    }
}
//static void getFiles(LPCTSTR path, vector<pair<LPCTSTR,int> > &filesPathVector)
//{
//	struct _tfinddata64_t c_file;
//	intptr_t hFile;
//	TCHAR root[MAX_PATH];
//	string p;
//	hFile=_tfindfirst64(p.assign(path).append("/*.*").c_str(),&c_file);
//	if( hFile == -1)
//		return;
//	static int auto_label = 1;
//	int label=0;
//	p.clear();
//	FILE *fp = fopen(p.assign(path).append("/label.txt").c_str(),"r");
//	if(fp==NULL)
//	{
//		label = auto_label++;
//		cout<<path<<" label:"<<label<<endl;
//	}
//	else
//	{
//		fscanf(fp,"%d",&label);
//		if(auto_label<label) auto_label = label + 1;
//	}
//
//	//search all files recursively.
//	do
//	{
//		if(_tcslen(c_file.name)==1&&c_file.name[0]==_T('.')
//			||_tcslen(c_file.name)==2&&c_file.name[0]==_T('.')&&c_file.name[1]==_T('.'))
//			continue;
//		TCHAR *fullPath =new TCHAR[MAX_PATH];
//		_tcscpy(fullPath,path);
//		_tcscat(fullPath,_T("\\"));
//		_tcscat(fullPath,c_file.name);
//		if(c_file.attrib&_A_SUBDIR)
//		{
//			getFiles(fullPath,filesPathVector);
//		}
//		else
//		{
//			if(strstr(c_file.name,"png")!=NULL||strstr(c_file.name,"bmp")!=NULL||strstr(c_file.name,"JPG")!=NULL
//				||strstr(c_file.name,"jpg")!=NULL||strstr(c_file.name,"tif")!=NULL)
//			//if(strstr(c_file.name,"yml")!=NULL)
//				filesPathVector.push_back(make_pair(fullPath,label));
//		}
//	}
//	while( _tfindnext64( hFile, &c_file ) == 0);
//	//close search handle
//	_findclose(hFile);
//}

int _tmain(int argc, char* argv[])
{
	char * filename = new char[100];
	char folder[]="G:\\CASIA-WebFace";
	vector<pair<string, int> > files;
	string p;
	FILE *fp = fopen(p.assign(folder).append("/list.txt").c_str(),"r");
	if(fp==NULL)
	{
		fp = fopen(p.assign(folder).append("/list.txt").c_str(),"w");
		getFiles(folder,files);
		int fileSize=files.size();
		for(int f=0;f<fileSize;f++){
			fprintf(fp,"%s %d\n",files[f].first.c_str(),files[f].second);
		}
		
	}
	else
	{
		char imgpath[256];
		int label;
		while(!feof(fp))
		{
			fscanf(fp,"%s %d",imgpath,&label);
			files.push_back( make_pair(imgpath,label) );
		}
	}
	fclose(fp);
	int fileSize=files.size();
	int * pResults = NULL; 


	for(int f=0;f<fileSize;f++){
		Mat gray = imread(files[f].first,0);
		imshow("original",gray);
		
	/////////////////////////////////////////////
	//// frontal face detection 
	//// it's fast, but cannot detect side view faces
	////////////////////////////////////////////
	////!!! The input image must be a gray one (single-channel)
	////!!! DO NOT RELEASE pResults !!!
		//pResults = facedetect_frontal((unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, gray.step,
		//1.2f, 2,  24);
	//printf("%d frontal faces detected.\n", (pResults ? *pResults : 0));
	////print the detection results
	//for(int i = 0; i < (pResults ? *pResults : 0); i++)
	//{
	//	short * p = ((short*)(pResults+1))+6*i;
	//	int x = p[0];
	//	int y = p[1];
	//	int w = p[2];
	//	int h = p[3];
	//	int neighbors = p[4];
	//	imshow("face",gray(Rect(x,y,w,h)));
	//	printf("face_rect=[%d, %d, %d, %d], neighbors=%d\n", x,y,w,h,neighbors);
	//}

	///////////////////////////////////////////
	// multiview face detection 
	// it can detection side view faces, but slower than the frontal face detection.
	//////////////////////////////////////////
	//!!! The input image must be a gray one (single-channel)
	//!!! DO NOT RELEASE pResults !!!
	pResults = facedetect_multiview((unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, gray.step,
		1.2f, 4, 24);
	printf("%d faces detected.\n", (pResults ? *pResults : 0));
	
	//print the detection results
	for(int i = 0; i < (pResults ? *pResults : 0); i++)
	{
		short * p = ((short*)(pResults+1))+6*i;
		int x = p[0];
		int y = p[1];
		int w = p[2];
		int h = p[3];
		int neighbors = p[4];
		int angle = p[5];
		imshow("face",gray(Rect(x,y,w,h)));
		printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x,y,w,h,neighbors, angle);
		//waitKey(1);
	}
	if(pResults==NULL)
	{
		waitKey(0);
	}
	else
	{
		waitKey(0);
	}
	}

	return 0;
}

