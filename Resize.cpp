#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include<algorithm>
#include <memory>


using namespace cv;
using namespace std;




//1724��
void InterNearestResize(const Mat &src, Mat &dst, Size &dsize, double fx = 0.0, double fy = 0.0)//ԭʼͼ���Լ����ű���
{
	
	Size ssize = src.size();//��ȡ�����С
	CV_Assert(ssize.area() > 0);//��֤����ĳ�������0

	if (!dsize.area())//���dsizeΪ(0,0)
	{
		dsize = Size(saturate_cast<int>(src.cols*fx),//satureate_cast��ֹ�������
			saturate_cast<int>(src.rows*fy));
		
		CV_Assert(dsize.area());
	}
	else
	{
		fx = (double)dsize.width / src.cols;//Size�еĿ�ߺ�mat�е��������෴��
		fy = (double)dsize.height / src.rows;
	}
	
	dst.create(Size(src.cols*fy,src.rows*fx), src.type());//��dst�����С��Ϊ��Ҫ�ĳߴ�
	
	fx = 1.0 / fx;
	fy = 1.0 / fy;
	uchar *ps = src.data;
	uchar *pd = dst.data;
	int channels = src.channels(),x,y,out,in;

	for (int row = 0; row < dst.rows; row++)
	{
		x = cvFloor(row * fx);
		for (int col = 0; col < dst.cols; col++)
		{
			y = cvFloor(col * fy);
			for (int c = 0; c < channels; c++)
			{
				out = (row * dst.cols + col) * channels + c;
				in = (x * src.cols + y) * channels + c;
				pd[out] = ps[in];
			}

			//dst.at<Vec3b>(row, col) = src.at<Vec3b>(x, y);
		}
	}
}

//(X, Y) = (1 - u)(1 - v)(x, y) + (u - 1)v(x, y + 1) + u(v - 1)(x + 1, y) + (uv)(x, y)
void InterLinerResize(const Mat &src, Mat &dst, Size &dsize, double fx = 0.0, double fy = 0.0)//˫���Բ�ֵ
{
	Size ssize = src.size();//��ȡ�����С
	CV_Assert(ssize.area() > 0);//��֤����ĳ�������0

	if (!dsize.area())//���dsizeΪ(0,0)
	{
		dsize = Size(saturate_cast<int>(src.cols*fx),//satureate_cast��ֹ�������
			saturate_cast<int>(src.rows*fy));

		CV_Assert(dsize.area());
	}
	else
	{
		fx = (double)dsize.width / src.cols;//Size�еĿ�ߺ�mat�е��������෴��
		fy = (double)dsize.height / src.rows;
	}

	dst.create(dsize, src.type());

	double ifx = 1. / fx;
	double ify = 1. / fy;

	uchar* dp = dst.data;
	uchar* sp = src.data;

	int iWidthSrc = src.cols;//��(������
	int iHiehgtSrc = src.rows;//��(����)
	int channels = src.channels();
	short cbufy[2];
	short cbufx[2];

	for (int row = 0; row < dst.rows; row++)
	{
		float fy = (float)((row + 0.5) * ify - 0.5);
		int sy = cvFloor(fy);//��������
		fy -= sy;//С������
		sy = std::min(sy, iHiehgtSrc - 2);
		sy = std::max(0, sy);

		cbufy[0] = cv::saturate_cast<short>((1.f - fy) * 2048);//1-u
		cbufy[1] = 2048 - cbufy[0];//u

		for (int col = 0; col < dst.cols; col++)
		{
			float fx = (float)((col + 0.5) * ifx - 0.5);
			int sx = cvFloor(fx);
			fx -= sx;

			if (sx < 0) 
			{
				fx = 0, sx = 0;
			}
			if (sx >= iWidthSrc - 1) 
			{
				fx = 0, sx = iWidthSrc - 2;
			}

			cbufx[0] = cv::saturate_cast<short>((1.f - fx) * 2048);//1-v
			cbufx[1] = 2048 - cbufx[0];//v

			for (int k = 0; k < src.channels(); ++k)
			{
				dp[(row * dst.cols + col) * channels + k] =
					(
					sp[ ( sy * src.cols + sx ) * channels + k] * cbufx[0] * cbufy[0] +
					sp[((sy + 1) * src.cols + sx) * channels + k] * cbufx[0] * cbufy[1] +
					sp[(sy * src.cols + (sx + 1)) * channels + k] * cbufx[1] * cbufy[0] +
					sp[((sy + 1) * src.cols + (sx + 1)) * channels + k] * cbufx[1] * cbufy[1]
					) >> 22;//2048*2048
			}
		}
	}

}

void Copy(Mat src, Mat &dst)//��src copy�� dst
{
	for (int row = 0; row < src.rows; row++)
	{
		//uchar *p = dst.ptr<uchar>(row);
		//uchar *s = src.ptr<uchar>(row);
		for (int col = 0; col < src.cols; col++)
		{
			//p[col] = s[col];
			dst.at<Vec3b>(row, col)[0] = src.at<Vec3b>(row, col)[0];
			dst.at<Vec3b>(row, col)[1] = src.at<Vec3b>(row, col)[1];
			dst.at<Vec3b>(row, col)[2] = src.at<Vec3b>(row, col)[2];
		}
	}
}


int main()
{
	ios::sync_with_stdio(false);
	Mat src = imread("src.jpg", IMREAD_UNCHANGED),out;//870*665


	InterLinerResize(src, out, Size(0, 0), 0.7, 0.5);
	
	
	imshow("src", src);
	//Copy(src, out);
	//InterNearestResize(src, out, Size(1000, 960));
	//InterLinerResize(src, out, Size(1080, 960));
	//resize(src, out,Size(1080,960),0);

	imshow("out", out);

	waitKey();
	
	system("pause");
	
	return 0;
}