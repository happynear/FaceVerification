// MatAlignment.cpp : 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"

#include <io.h>
#include <direct.h>
#include <string>
#include <iomanip>
#include <Windows.h>
#include <AtlBase.h>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include "facedetect-dll.h"
#include "mex.h"

using namespace dlib;
using namespace std;
#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

frontal_face_detector detector;
shape_predictor sp;
cv::Mat image;

// Do CHECK and throw a Mex error if check fails
inline void mxCHECK(bool expr, const char* msg) {
  if (!expr) {
    mexErrMsgTxt(msg);
  }
}
inline void mxERROR(const char* msg) { mexErrMsgTxt(msg); }

// Check if a file exists and can be opened
void mxCHECK_FILE_EXIST(const char* file) {
  std::ifstream f(file);
  if (!f.good()) {
    f.close();
    std::string msg("Could not open file ");
    msg += file;
    mxERROR(msg.c_str());
  }
  f.close();
}

void InitModel(MEX_ARGS)
{
	mxCHECK(nrhs == 1 && mxIsChar(prhs[0]),
      "Usage: MatAlignment('init_model', model_file)");
    char* model_file = mxArrayToString(prhs[0]);
	mxCHECK_FILE_EXIST(model_file);
	detector = get_frontal_face_detector();
    deserialize(model_file) >> sp;
	mxFree(model_file);
}

// Copy matlab array to Blob data or diff
static void ReadMat(const mxArray* mx_mat) {
	const size_t* mat_size = mxGetDimensions(mx_mat);
	image = cv::Mat(mat_size[1],mat_size[0],CV_8UC1);
	const unsigned char* mat_mem_ptr = reinterpret_cast<const unsigned char*>(mxGetData(mx_mat));
	memcpy(image.data, mat_mem_ptr, sizeof(unsigned char) * mxGetNumberOfElements(mx_mat));
	image = image.t();
}

// Copy matlab array to Blob data or diff
static mxArray* WriteMat(const cv::Mat& opencv_mat) {
	std::vector<mwSize> dims(2);
		
	dims[0] = static_cast<mwSize>(opencv_mat.rows);
	dims[1] = static_cast<mwSize>(opencv_mat.cols);
		
	mxArray* mx_mat =
		mxCreateNumericArray(2, dims.data(), mxSINGLE_CLASS, mxREAL);
	float* mat_mem_ptr = reinterpret_cast<float*>(mxGetData(mx_mat));
	memcpy(mat_mem_ptr, opencv_mat.data, sizeof(float) * dims[0] * dims[1]);
	return mx_mat;
}

static void Alignment(MEX_ARGS)
{
	mxCHECK(nrhs == 1 &&  mxIsUint8(prhs[0]),
      "Usage: MatAlignment('alignment', image)");
	ReadMat(prhs[0]);
	mexPrintf("Read Mat to OpenCV done.\n");
	dlib::cv_image<unsigned char> cv_img(image);
	//imwrite("e:\\test.bmp",image);
			
	array2d<unsigned char> img;
	assign_image(img,cv_img);
	std::vector<rectangle> dets = detector(img);
	if(dets.size()>1)
	{
		int maxArea = 0,p = 0;
		for (int i = 0;i<dets.size();i++)
		{
			double vote = dets[i].width()*dets[i].height();
			vote -= abs(image.cols/2 - (dets[i].left() + dets[i].right())/2) * abs(image.rows/2 - (dets[i].top() + dets[i].bottom())/2);
			if(vote>maxArea)
			{
				maxArea = dets[i].width()*dets[i].height();
				p = i;
			}
		}
		if(p!=0)
		{
			dets.erase(dets.begin(),dets.begin()+p-1);
		}
	}
	mexPrintf("%d face detected.\n",dets.size());
	std::vector<full_object_detection> shapes;
	if(dets.size()>0)
	{
        full_object_detection shape = sp(img, dets[0]);
        shapes.push_back(shape);
    }
	else
	{
		size_t dims[1]={1};
		mxArray* mx_mat =
			mxCreateNumericArray(1, dims, mxLOGICAL_CLASS, mxREAL);
		bool* mat_mem_ptr = reinterpret_cast<bool*>(mxGetData(mx_mat));
		mat_mem_ptr[0] = 1;
		plhs[0] = mx_mat;
	}
	if(shapes.size()>0)
	{
		dlib::array<array2d<rgb_pixel> > face_chips;
		extract_image_chips(img, get_face_chip_details(shapes,100UL,0.5785), face_chips);
			
		array2d<unsigned char> gray_face;
		assign_image(gray_face,face_chips[0]);
		mexPrintf("Face alignment done.\n");
		cv::Mat face = toMat (gray_face);
		face.convertTo(face,CV_32F);

		plhs[0] = WriteMat(face);
		mexPrintf("Convert to mat done.\n");
	}
}

double OverLap(rectangle &r1,rectangle &r2)
{
	rectangle overlap = rectangle(max(r1.left(),r2.left()),max(r1.top(),r2.top()),min(r1.right(),r2.right()),min(r1.bottom(),r2.bottom()));
	return (double)overlap.area() / (double)(r1.area()+r2.area()-overlap.area());
}

/** -----------------------------------------------------------------
 ** Available commands.
 **/
struct handler_registry {
  string cmd;
  void (*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "init_model",         InitModel      },
  { "alignment",          Alignment      },
  // The end.
  { "END",                NULL            },
};

void mexFunction(MEX_ARGS) {
  mexLock();  // Avoid clearing the mex file.
  mxCHECK(nrhs > 0, "Usage: caffe_(api_command, arg1, arg2, ...)");
  {// Handle input command
  char* cmd = mxArrayToString(prhs[0]);
  bool dispatched = false;
  // Dispatch to cmd handler
  for (int i = 0; handlers[i].func != NULL; i++) {
    if (handlers[i].cmd.compare(cmd) == 0) {
      handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
      dispatched = true;
      break;
    }
  }
  if (!dispatched) {
    ostringstream error_msg;
    error_msg << "Unknown command '" << cmd << "'";
    mxERROR(error_msg.str().c_str());
  }
  mxFree(cmd);
  }
}