#include "arcface.h"

Arcface::Arcface(string model_folder)
{
    string param_file = model_folder + "/mobilefacenet.param";
    string bin_file = model_folder + "/mobilefacenet.bin";

    this->net.load_param(param_file.c_str());
    this->net.load_model(bin_file.c_str());
}

Arcface::~Arcface()
{
    this->net.clear();
}

vector<float> Arcface::getFeature(ncnn::Mat img)
{
    vector<float> feature;
    ncnn::Mat in = resize(img, 112, 112);
    in = bgr2rgb(in);
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("fc1", out);
    feature.resize(this->feature_dim);
    for (int i = 0; i < this->feature_dim; i++)
        feature[i] = out[i];
    normalize(feature);
    return feature;
}

void Arcface::normalize(vector<float> &feature)
{
    float sum = 0;
    for (auto it = feature.begin(); it != feature.end(); it++)
        sum += (float)*it * (float)*it;
    sum = sqrt(sum);
    for (auto it = feature.begin(); it != feature.end(); it++)
        *it /= sum;
}

ncnn::Mat preprocess(ncnn::Mat img, FaceInfo info)
{
    int image_w = 112; //96 or 112
    int image_h = 112;

    float dst[10] = {30.2946, 65.5318, 48.0252, 33.5493, 62.7299,
                     51.6963, 51.5014, 71.7366, 92.3655, 92.2041};

    if (image_w == 112)
        for (int i = 0; i < 5; i++)
            dst[i] += 8.0;

    float src[10];
    for (int i = 0; i < 5; i++)
    {
        src[i] = info.landmark[2 * i];
        src[i + 5] = info.landmark[2 * i + 1];
    }

    float M[6];
    getAffineMatrix(src, dst, M);
    ncnn::Mat out;
    warpAffineMatrix(img, out, M, image_w, image_h);
    return out;
}

float calcSimilar(std::vector<float> feature1, std::vector<float> feature2)
{
    //assert(feature1.size() == feature2.size());
    float sim = 0.0;
    for (int i = 0; i < feature1.size(); i++)
        sim += feature1[i] * feature2[i];
    return sim;
}
