#ifndef _POSTPROCESS_H_
#define _POSTPROCESS_H_

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <vector>

#include <opencv2/highgui.hpp>

typedef signed char int8_t;
typedef unsigned int uint32_t;


typedef struct
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int classId;
    float mask[32];
} DetectRect;

// yolov8
class GetResultRectYolov8
{
public:
    GetResultRectYolov8();

    ~GetResultRectYolov8();

    int GenerateMeshgrid();

    int GetConvDetectionResult(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects, cv::Mat &SegMask);

    float sigmoid(float x);

private:
    std::vector<float> meshgrid;

    const int class_num = 1;
    int headNum = 3;

    int input_w = 640;
    int input_h = 640;
    int strides[3] = {8, 16, 32};
    int mapSize[3][2] = {{80, 80}, {40, 40}, {20, 20}};
    
    int maskNum = 32;
    int mask_seg_w = 160;
    int mask_seg_h = 160;

    float nmsThresh = 0.45;
    float objectThresh = 0.5;

    std::vector<cv::Vec3b> ColorLists = {cv::Vec3b(000, 000, 255),
                                        cv::Vec3b(255, 128, 000),
                                        cv::Vec3b(255, 255, 000),
                                        cv::Vec3b(000, 255, 000),
                                        cv::Vec3b(000, 255, 255),
                                        cv::Vec3b(255, 000, 000),
                                        cv::Vec3b(128, 000, 255),
                                        cv::Vec3b(255, 000, 255),
                                        cv::Vec3b(128, 000, 000),
                                        cv::Vec3b(000, 128, 000)};

};

#endif
