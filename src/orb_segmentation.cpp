/*
 *
 *  Created on: Jan 21, 2017
 *      Author: Timo Sämann
 *
 *  The basis for the creation of this script was the classification.cpp example (caffe/examples/cpp_classification/classification.cpp)
 *
 *  This script visualize the semantic segmentation for your input image.
 *
 *  To compile this script you can use a IDE like Eclipse. To include Caffe and OpenCV in Eclipse please refer to
 *  http://tzutalin.blogspot.de/2015/05/caffe-on-ubuntu-eclipse-cc.html
 *  and http://rodrigoberriel.com/2014/10/using-opencv-3-0-0-with-eclipse/ , respectively
 *
 *
 */

//实验一
//单帧的特征检测和特征匹配


#define USE_OPENCV 1
#include "/home/ubuntu/caffe-segnet/include/caffe/caffe.hpp"

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#endif  // USE_OPENCV

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
//#include <chrono> //Just for time measurement. This library requires compiler and library support for the ISO C++ 2011 standard. This support is currently experimental in Caffe, and must be enabled with the -std=c++11 or -std=gnu++11 compiler options.


#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;
using namespace cv;

#define MATCH_DISTANCE 180
#define CLASS_PEOPLE 15

//Mat img prediction;
class Classifier 
{
 public:
  Classifier(const string& model_file,const string& trained_file);

  void Predict(const cv::Mat& img, cv::Mat& img_segment);

 private:
  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);

 private:
  boost::shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
};

Classifier::Classifier(const string& model_file,const string& trained_file)
{
  Caffe::set_mode(Caffe::GPU);

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)<< "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

void Classifier::Predict(const cv::Mat& img, cv::Mat& img_segment)
{
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,input_geometry_.height, input_geometry_.width);
  
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);
  
  struct timeval time;
  gettimeofday(&time, NULL); // Start Time
  long totalTime = (time.tv_sec * 1000) + (time.tv_usec / 1000);
  //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); //Just for time measurement

  net_->Forward();

  gettimeofday(&time, NULL);  //END-TIME
  totalTime = (((time.tv_sec * 1000) + (time.tv_usec / 1000)) - totalTime);
  std::cout << "Processing time = " << totalTime << " ms" << std::endl;

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
    
  //获得只有人的灰度图像
  // 仿照博客: caffe之特征图可视化及特征提取 https://blog.csdn.net/zxj942405301/article/details/71195267
  //将测试图片经过最后一个卷积层的特征图可视化
  //这一层的名字叫做 prob
  string blobName="prob";    
  assert(net_->has_blob(blobName));    //为免出错，我们必须断言，网络中确实有名字为blobName的特征图  
  boost::shared_ptr<Blob<float> >  conv1Blob=net_->blob_by_name(blobName); 
  //为了可视化，仍然需要归一化到0~255    
  //下面的代码，跟上一篇博客中是一样的  
  float maxValue=-10000000,minValue=10000000;   
  const float* tmpValue=conv1Blob->cpu_data();       
  for(int i=0;i<conv1Blob->count();i++)
  {          
      maxValue=std::max(maxValue,tmpValue[i]);          
      minValue=std::min(minValue,tmpValue[i]);      
  }  
  int width=conv1Blob->shape(2);  //响应图的宽度  
  int height=conv1Blob->shape(3);  //响应图的高度   
  //获取第15类 "人"
  Mat img_i(width,height,CV_8UC1);
  for(int i=0;i<height;i++)
  {               
      for(int j=0;j<width;j++)
      {                        
      float value=conv1Blob->data_at(0,CLASS_PEOPLE,i,j);  
      img_i.at<uchar>(i,j)=(value-minValue)/(maxValue-minValue)*255;  
      }       
  }
  //转化成与原图片相同的格式
  resize(img_i,img_i,Size(640,480)); 
  //将结果输出
  img_segment = img_i.clone();
}

void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) 
{
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels)
{
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc,char *argv[]) 
{
  //初始化
  ::google::InitGoogleLogging(argv[0]);

  //参数校验
  if ( argc!=5 )
  {
    std::cout<<" usage: ./bin/orb_segmentation *.prototxt *.caffemodel image1 image2 "<<std::endl;
    return 1;
  }
  
  //加载参数文件的模型文件  
  string prototxt_file = argv[1];
  string caffemodel_file = argv[2];
  
  //分类器对象初始化
  Classifier classifier(prototxt_file, caffemodel_file);
  
  //读入两帧图像
  cv::Mat img_1 = imread(argv[3],1);
  cv::Mat img_2 = imread(argv[4],1);
  cv::Mat img_1_seg = img_1.clone();
  cv::Mat img_2_seg = img_2.clone();
  
  //保存原始图像
  imwrite("./result/原图1.png",img_1);
  imwrite("./result/原图2.png",img_2);
  
  //传统特征匹配
  std::cout<<" ########### 传统特征提取和匹配 ###########"<<std::endl;
  //定义存储关键点的容器
  std::vector<cv::KeyPoint> keypoints1,keypoints2;
  //定义存储描述子的容器
  cv::Mat descriptors1,descriptors2;
  //orb特征检测对象
  cv::ORB orb_obj( 1000, 1.2f, 1, 31, 0, 2, ORB::HARRIS_SCORE, 31 );
  //特征检测
  orb_obj.detect(img_1,keypoints1);
  orb_obj.detect(img_2,keypoints2);
  std::cout<<" 两帧图像分别检测到 "<<keypoints1.size()<<" , "<<keypoints2.size()<<" 个特征点;"<<std::endl;
  //特征点绘制  
  Mat image1_out;
  Mat image2_out;
  drawKeypoints(img_1,keypoints1,image1_out,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
  drawKeypoints(img_2,keypoints2,image2_out,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
  imshow("筛选前图一提取到的ORB特征点",image1_out);
  imshow("筛选前图二提取到的ORB特征点",image2_out);
  //保存传统特征提取结果
  imwrite("./result/筛选图一提取到的ORB特征点.png",image1_out);
  imwrite("./result/筛选前图二提取到的ORB特征点.png",image2_out);
  waitKey(1);
  //描述子计算
  orb_obj.compute(img_1,keypoints1,descriptors1);
  orb_obj.compute(img_2,keypoints2,descriptors2);
  //特征匹配
  //定义描述子匹配对象
  BFMatcher orb_matcher(NORM_L2);  
  //定义存储匹配结果的容器
  vector<DMatch> matches;
  vector<DMatch> good_matches;
  orb_matcher.match(descriptors1,descriptors2,matches);
  //根据距离进行筛选
  double max_dist=0,min_dist=1000;
  for( int i=0;i<matches.size();i++)
  {
    double distance = matches[i].distance;
    if(distance<min_dist)
      min_dist=distance;
    if(distance>max_dist)
      max_dist=distance;
  }  
  for( int j=0;j<matches.size();j++)
  {
      if(matches[j].distance <= MATCH_DISTANCE )
      //if(matches[j].distance <= max( 2*min_dist,30.0))
      good_matches.push_back(matches[j]);
  }  
  //绘制匹配的点对
  Mat image_good_matches;
  drawMatches(img_1,keypoints1,img_2,keypoints2,good_matches,image_good_matches);
  imshow("筛选前图一和图二的ORB匹配结果",image_good_matches);
  //保存原始匹配结果
  imwrite("./result/筛选前图一和图二的ORB匹配结果.png",image_good_matches);
  std::cout<<" 筛选前匹配到 "<<good_matches.size()<<std::endl;
  waitKey(1);
  
  
  //语义分割后的特征提取和匹配
  std::cout<<" ####### 语义分割后的特征提取和匹配 #######"<<std::endl;
  //特征剔除
  //生成分割图像img_segment
  cv::Mat img_segment_1,img_segment_2;
  classifier.Predict(img_1, img_segment_1);
  classifier.Predict(img_2, img_segment_2);
  //分割图像显示
  namedWindow("用于动态点剔除的图像1", 0); 
  namedWindow("用于动态点剔除的图像2", 0);
  cv::imshow( "用于动态点剔除的图像1", img_segment_1);
  cv::imshow( "用于动态点剔除的图像2", img_segment_2);
  //保存用于特征剔除的图像
  imwrite("./result/用于动态点剔除的图像1.png",img_segment_1);
  imwrite("./result/用于动态点剔除的图像2.png",img_segment_2);
  //剔除
  vector<KeyPoint> keypoints_seg_1,keypoints_seg_2;
  cv::Mat descriptors_seg_1,descriptors_seg_2;
  for ( int i=0;i<keypoints1.size();i++ )
  {
     if ( img_segment_1.at<unsigned char>(keypoints1[i].pt.y,keypoints1[i].pt.x) == 0)
     {
       keypoints_seg_1.push_back(keypoints1[i]);
      }
  }
  for ( int i=0;i<keypoints2.size();i++ )
  {
     if ( img_segment_2.at<unsigned char>(keypoints2[i].pt.y,keypoints2[i].pt.x) == 0)
     {
       keypoints_seg_2.push_back(keypoints2[i]);
      }
  }
  std::cout<<" 筛选后分别有 "<<keypoints_seg_1.size()<<","<<keypoints_seg_2.size()<<"个特征点;"<<std::endl;
  //特征点绘制  
  Mat image_seg_1_out;
  Mat image_seg_2_out;
  drawKeypoints(img_1,keypoints_seg_1,image_seg_1_out,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
  drawKeypoints(img_2,keypoints_seg_2,image_seg_2_out,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
  imshow("筛选后图一提取到的ORB特征点",image_seg_1_out);
  imshow("筛选后图二提取到的ORB特征点",image_seg_2_out);
  //保存特征剔除后的图像
  imwrite("./result/筛选后图一提取到的ORB特征点.png",image_seg_1_out);
  imwrite("./result/筛选后图二提取到的ORB特征点.png",image_seg_1_out);
  waitKey(1);
  //描述子计算
  orb_obj.compute(img_1,keypoints_seg_1,descriptors_seg_1);
  orb_obj.compute(img_2,keypoints_seg_2,descriptors_seg_2);
  vector<DMatch> matches_seg;
  vector<DMatch> good_matches_seg;
  orb_matcher.match(descriptors_seg_1,descriptors_seg_2,matches_seg);
  //根据距离进行筛选
  max_dist=0;
  min_dist=1000;
  for( int i=0;i<matches_seg.size();i++)
  {
    double distance = matches_seg[i].distance;
    if(distance<min_dist)
      min_dist=distance;
    if(distance>max_dist)
      max_dist=distance;
  }  
  for( int j=0;j<matches_seg.size();j++)
  {
      if(matches_seg[j].distance <= MATCH_DISTANCE )
      //if(matches[j].distance <= max( 2*min_dist,30.0))
      good_matches_seg.push_back(matches_seg[j]);
  }  
  //绘制匹配的点对
  Mat image_good_matches_seg;
  drawMatches(img_1_seg,keypoints_seg_1,img_2_seg,keypoints_seg_2,good_matches_seg,image_good_matches_seg);
  imshow("筛选后图一和图二的ORB匹配结果",image_good_matches_seg);
  //保存特征剔除后的匹配结果
  imwrite("./result/筛选后图一和图二的ORB匹配结果.png",image_good_matches_seg);
  std::cout<<" 筛选后匹配到 "<<good_matches_seg.size()<<std::endl;
  waitKey(0);
  destroyAllWindows();
  

}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV


