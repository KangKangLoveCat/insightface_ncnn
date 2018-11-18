#ifndef MTCNN_H
#define MTCNN_H

#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include "net.h"
#include "base.h"

using namespace std;

class MtcnnDetector {
public:
    MtcnnDetector(string model_folder = ".");
    ~MtcnnDetector();
    vector<FaceInfo> Detect(ncnn::Mat img);
private:
    float minsize = 20;
    float threshold[3] = {0.6f, 0.7f, 0.8f};
    float factor = 0.709f;
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {0.0078125f, 0.0078125f, 0.0078125f};
    ncnn::Net Pnet;
    ncnn::Net Rnet;
    ncnn::Net Onet;
    ncnn::Net Lnet;
    vector<FaceInfo> Pnet_Detect(ncnn::Mat img);
    vector<FaceInfo> Rnet_Detect(ncnn::Mat img, vector<FaceInfo> bboxs);
    vector<FaceInfo> Onet_Detect(ncnn::Mat img, vector<FaceInfo> bboxs);
    void Lnet_Detect(ncnn::Mat img, vector<FaceInfo> &bboxs);
    vector<FaceInfo> generateBbox(ncnn::Mat score, ncnn::Mat loc, float scale, float thresh);
    void doNms(vector<FaceInfo> &bboxs, float nms_thresh, string mode);
    void refine(vector<FaceInfo> &bboxs, int height, int width, bool flag = false);
};

#endif
