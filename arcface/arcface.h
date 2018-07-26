#ifndef ARCFACE_H
#define ARCFACE_H

#include <cmath>
#include <vector>
#include <string>
#include "net.h"
#include "base.h"

using namespace std;

ncnn::Mat preprocess(ncnn::Mat img, FaceInfo info);

float calcSimilar(std::vector<float> feature1, std::vector<float> feature2);


class Arcface {

public:
    Arcface(string model_folder = ".");
    ~Arcface();
    vector<float> getFeature(ncnn::Mat img);

private:
    ncnn::Net net;

    const int feature_dim = 128;

    void normalize(vector<float> &feature);
};

#endif
