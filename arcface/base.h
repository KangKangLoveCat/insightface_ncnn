#ifndef BASE_H
#define BASE_H
#include <cmath>
#include <cstring>
#include "net.h"

typedef struct FaceInfo {
    float score;
    int x[2];
    int y[2];
    float area;
    float regreCoord[4];
    int landmark[10];
} FaceInfo;

ncnn::Mat resize(ncnn::Mat src, int w, int h);

ncnn::Mat bgr2rgb(ncnn::Mat src);

ncnn::Mat rgb2bgr(ncnn::Mat src);

void getAffineMatrix(float* src_5pts, const float* dst_5pts, float* M);

void warpAffineMatrix(ncnn::Mat src, ncnn::Mat &dst, float *M, int dst_w, int dst_h);

#endif
