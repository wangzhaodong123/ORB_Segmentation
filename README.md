# ORB_Segmenttation
使用语义分割将图像中的人分离出来,将原始图像中提取的ORB特征点落在人身上的剔除(用于SLAM的位姿估计).
# 参考
1,caffe之特征图可视化及特征提取

博客链接:https://blog.csdn.net/zxj942405301/article/details/71195267

2,caffe-segnet-cudnn5的test_segmentation.cpp

链接:https://github.com/TimoSaemann/caffe-segnet-cudnn5/tree/master/examples/SegNet_with_C%2B%2B
# 编译运行
1,首先安装caffe-segnet-cudnn5

github地址:https://github.com/TimoSaemann/caffe-segnet-cudnn5

2,cd ORB_Segmenttation

  mkdir build
  
  cd build
  
  cmake ..
  
  make
  
3,cd ORB_Segmenttation
  
  ./bin/orb_segmentation models/segnet_pascal.prototxt models/segnet_pascal.caffemodel images/image_1.png images/image_2.png
  
4,输出结果将保存在result/文件夹下
 
