#include "visual_odom.h"

#include <string>
#include <fstream>
#include <iomanip>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"


#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

const int kMinNumFeature = 2000;


//using namespace cv;
using namespace std;




visual_odom::visual_odom(Pinhole* cam):cam_(cam)
{
	focal_ = cam_->fx();
	pp_ = cv::Point2d(cam_->cx(), cam_->cy());
	frame_stage_ = STAGE_FIRST_FRAME;

}

void visual_odom::update(const cv::Mat& img, int frame_id)
{
	//if (img.empty() || img.type() != CV_8UC1 || img.cols != cam_->width() || img.rows != cam_->height())
//throw std::runtime_error("Frame: provided image has not the same size as the camera model or image is not grayscale");

	new_frame_ = img;
	bool res = true;
	if (frame_stage_ == STAGE_DEFAULT_FRAME)
		res = processFrame(frame_id);
	else if (frame_stage_ == STAGE_SECOND_FRAME)
		res = processSecondFrame();
	else if (frame_stage_ == STAGE_FIRST_FRAME)
		res = processFirstFrame();

	last_frame_ = new_frame_;

}


bool visual_odom::processFirstFrame()
{
	featureDetection(new_frame_, px_ref_);
	frame_stage_ = STAGE_SECOND_FRAME;
		return true;

}


bool visual_odom::processSecondFrame()
{
	featureTracking(last_frame_, new_frame_, px_ref_, px_cur_, status);

	cv::Mat E, R, t, mask; 

	E = cv::findEssentialMat(px_cur_,px_ref_, focal_, pp_, cv::RANSAC, 0.999,1.0, mask );

	cv::recoverPose(E, px_cur_, px_ref_,  R, t, focal_, pp_, mask);

	cur_R_ = R.clone();
	cur_t_ = t.clone();

	frame_stage_ = STAGE_DEFAULT_FRAME;
	px_ref_ = px_cur_;

	return true;

}


bool visual_odom::processFrame(int frame_id)
{
	double absolutescale = 1.0;
	featureTracking(last_frame_, new_frame_, px_ref_, px_cur_,status);

	cv::Mat E, R, t, mask; 

	E = cv::findEssentialMat(px_cur_,px_ref_, focal_, pp_, cv::RANSAC, 0.999,1.0, mask );

	cv::recoverPose(E, px_cur_, px_ref_,  R, t, focal_, pp_, mask);


	absolutescale = getAbsoluteScale(frame_id);

	if (absolutescale > 0.1) 
	{
		cur_t_ = cur_t_ + absolutescale*(cur_R_*t);
		cur_R_ = R*cur_R_;
	}

	if (px_ref_.size() < kMinNumFeature)
	{
		featureDetection(new_frame_, px_ref_);
		featureTracking(last_frame_, new_frame_, px_ref_, px_cur_, status);
	}

	px_ref_ = px_cur_;

	return true;



}


double visual_odom::getAbsoluteScale(int frame_id)
{
	std::string line;
	int i = 0;
	std::ifstream ground_truth("/home/abhilesh/Odometry/dataset/sequences/05/05.txt");
	double x = 0, y = 0, z = 0;
	double x_prev, y_prev, z_prev;
	if (ground_truth.is_open())
	{
		while ((std::getline(ground_truth, line)) && (i <= frame_id))
		{
			z_prev = z;
			x_prev = x;
			y_prev = y;
			std::istringstream in(line);
			for (int j = 0; j < 12; j++)  {
				in >> z;
				if (j == 7) y = z;
				if (j == 3)  x = z;
			}
			i++;
		}
		ground_truth.close();
	}

	else {
		std::cerr<< "Unable to open file";
		return 0;
	}

	return sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev));
}

//--------------------------------------------------------------------------------------------------------------------------------


int main(int argc, char *argv[])
{
	
	Pinhole *cam = new Pinhole(1241.0, 376.0,
		718.8560, 718.8560, 607.1928, 185.2157);
	visual_odom vo(cam);

	std::ofstream out("position.txt");


	char text[100];
	int font_face = cv::FONT_HERSHEY_PLAIN;
	double font_scale = 1;
	int thickness = 1;
	cv::Point text_org(10, 50);
	cv::namedWindow("Road facing camera", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);
	cv::Mat traj = cv::Mat::zeros(650,650, CV_8UC3);

	double x=0.0, y=0.0,z=0.0;
	for (int img_id = 0; img_id < 2760; ++img_id)
	{
		
		std::stringstream ss;
		ss <<  "/home/abhilesh/Odometry/dataset/sequences/05/image_1/"
			<< std::setw(6) << std::setfill('0') << img_id << ".png";

		cv::Mat img(cv::imread(ss.str().c_str(), 0));

		assert(!img.empty());
		cout<<img_id<<endl;

		
		vo.update(img, img_id);
		cv::Mat cur_t = vo.getCurrentT();
		if (cur_t.rows!=0)
		{
			x = cur_t.at<double>(0);
			y = cur_t.at<double>(1);
			z = cur_t.at<double>(2);
		}
		out << x << " " << y << " " << z << std::endl;

		int draw_x = int(x) + 300;
		int draw_y = int(z) + 150;
		cv::circle(traj, cv::Point(draw_x, draw_y), 1, CV_RGB(0, 255, 0), 2);

		cv::rectangle(traj, cv::Point(10, 30), cv::Point(580, 60), CV_RGB(0, 0, 0), CV_FILLED);
		sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", x, y, z);
		cv::putText(traj, text, text_org, font_face, font_scale, cv::Scalar::all(255), thickness, 8);

		cv::imshow("Road facing camera", img);
		cv::imshow("Trajectory", traj);

		cv::waitKey(1);
	}

	delete cam;
	out.close();
	getchar();
	return 0;
}