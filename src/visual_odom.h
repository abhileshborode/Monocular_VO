#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Pinhole
{
public:


Pinhole(double width, double height,
	double fx, double fy,
	double cx, double cy,
	double k1=0.0, double k2=0.0, double p1=0.0, double p2=0.0, double k3=0.0) :
	width_(width), height_(height),
	fx_(fx), fy_(fy), cx_(cx), cy_(cy),
	distortion_(fabs(k1) > 0.0000001)
{
	d_[0] = k1; d_[1] = k2; d_[2] = p1; d_[3] = p2; d_[4] = k3;
	
}

~Pinhole()
{}
	

	inline int width() const { return width_; }
	inline int height() const { return height_; }
	inline double fx() const { return fx_; };
	inline double fy() const { return fy_; };
	inline double cx() const { return cx_; };
	inline double cy() const { return cy_; };
	double k1()
	{
		return d_[0];
	}

	double k2()
	{
		return d_[1];	
	}

	double p1()
	{
		return d_[2];
	}

	double p2()
	{
		return d_[3];
	}
	double k3()
	{
		return d_[4];
	}

	private:
	double width_ , height_, fx_,fy_,cx_,cy_;
	bool distortion_;

	double d_[5];

};

class visual_odom
{
public:

	enum FrameStage 
	{
		STAGE_FIRST_FRAME,
		STAGE_SECOND_FRAME,
		STAGE_DEFAULT_FRAME
	};

	visual_odom(Pinhole* cam);


	virtual ~visual_odom()
	{

	}
	


	virtual bool processFirstFrame();
	
	virtual bool processSecondFrame();

	virtual bool processFrame(int frame_id);

	double getAbsoluteScale(int frame_id);


//	void featureTracking(cv::Mat img_1, cv::Mat img_2, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2, vector<uchar> status);

//	void featureDetection(cv::Mat img_1, vector<cv::Point2f>& points1);


void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)	{ 

//this function automatically gets rid of points for which tracking fails

  vector<float> err;					
  Size winSize=Size(21,21);																								
  TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)
     {  Point2f pt = points2.at(i- indexCorrection);
     	if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
     		  if((pt.x<0)||(pt.y<0))	{
     		  	status.at(i) = 0;
     		  }
     		  points1.erase (points1.begin() + (i - indexCorrection));
     		  points2.erase (points2.begin() + (i - indexCorrection));
     		  indexCorrection++;
     	}

     }

}


void featureDetection(Mat img_1, vector<Point2f>& points1)	{   //uses FAST as of now, modify parameters as necessary
  vector<KeyPoint> keypoints_1;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
  KeyPoint::convert(keypoints_1, points1, vector<int>());
}
	
	void update(const cv::Mat& img, int frame_id);

	
	cv::Mat getCurrentR() { return cur_R_; }
	
	cv::Mat getCurrentT() { return cur_t_; }



protected:
	FrameStage frame_stage_;                 
	Pinhole *cam_;                     
	cv::Mat new_frame_;                      
	cv::Mat last_frame_;                     

	cv::Mat cur_R_;
	cv::Mat cur_t_;

	std::vector<cv::Point2f> px_ref_;      
	std::vector<cv::Point2f> px_cur_;      
	std::vector<double> disparities_;     
	std::vector<uchar> status;
	double focal_;
	cv::Point2d pp_; 


};