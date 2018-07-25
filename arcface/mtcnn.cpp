#include "mtcnn.h"

MtcnnDetector::MtcnnDetector(string model_folder)
{
    vector<string> param_files = {
        model_folder + "/det1.param", 
        model_folder + "/det2.param",
        model_folder + "/det3.param",
        model_folder + "/det4.param"
    };
    vector<string> bin_files = {
        model_folder + "/det1.bin",
        model_folder + "/det2.bin",
        model_folder + "/det3.bin",
        model_folder + "/det4.bin"
    };
    this->Pnet.load_param(param_files[0].c_str());
    this->Pnet.load_model(bin_files[0].c_str());
    this->Rnet.load_param(param_files[1].c_str());
    this->Rnet.load_model(bin_files[1].c_str());
    this->Onet.load_param(param_files[2].c_str());
    this->Onet.load_model(bin_files[2].c_str());
    this->Lnet.load_param(param_files[3].c_str());
    this->Lnet.load_model(bin_files[3].c_str());
}

MtcnnDetector::~MtcnnDetector()
{
    this->Pnet.clear();
    this->Rnet.clear();
    this->Onet.clear();
    this->Lnet.clear();
}

vector<FaceInfo> MtcnnDetector::Detect(ncnn::Mat img)
{
    int img_w = img.w;
    int img_h = img.h;

    vector<FaceInfo> pnet_results = Pnet_Detect(img);
    doNms(pnet_results, 0.7, "union");
    refine(pnet_results, img_h, img_w, true);

    vector<FaceInfo> rnet_results = Rnet_Detect(img, pnet_results);
    doNms(rnet_results, 0.7, "union");
    refine(rnet_results, img_h, img_w, true);

    vector<FaceInfo> onet_results = Onet_Detect(img, rnet_results);
    refine(onet_results, img_h, img_w, false);
    doNms(onet_results, 0.7, "min");

    Lnet_Detect(img, onet_results);

    return onet_results;
}

vector<FaceInfo> MtcnnDetector::Pnet_Detect(ncnn::Mat img)
{
    vector<FaceInfo> results;
    int img_w = img.w;
    int img_h = img.h;
    float minl = img_w < img_h ? img_w : img_h;
    double scale = 12.0 / this->minsize;
    minl *= scale;
    vector<double> scales;
    while (minl > 12)
    {
        scales.push_back(scale);
        minl *= this->factor;
        scale *= this->factor;
    }
    for (auto it = scales.begin(); it != scales.end(); it++)
    {
        scale = (double)(*it);
        int hs = (int) ceil(img_h * scale);
        int ws = (int) ceil(img_w * scale);
        ncnn::Mat in = resize(img, ws, hs);
        in.substract_mean_normalize(this->mean_vals, this->norm_vals);
        ncnn::Extractor ex = Pnet.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score;
        ncnn::Mat location;
        ex.extract("prob1", score);
        ex.extract("conv4_2", location);
        vector<FaceInfo> bboxs = generateBbox(score, location, *it, this->threshold[0]);
        doNms(bboxs, 0.5, "union");
        results.insert(results.end(), bboxs.begin(), bboxs.end());
    }
    return results;
}

vector<FaceInfo> MtcnnDetector::Rnet_Detect(ncnn::Mat img, vector<FaceInfo> bboxs)
{
    vector<FaceInfo> results;

    int img_w = img.w;
    int img_h = img.h;

    for (auto it = bboxs.begin(); it != bboxs.end(); it++)
    {
        ncnn::Mat img_t;
        copy_cut_border(img, img_t, it->y[0], img_h - it->y[1], it->x[0], img_w - it->x[1]);
        ncnn::Mat in = resize(img_t, 24, 24);
        in.substract_mean_normalize(this->mean_vals, this->norm_vals);
        ncnn::Extractor ex = Rnet.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score, bbox;
        ex.extract("prob1", score);
        ex.extract("conv5_2", bbox);
        if ((float)score[1] > threshold[1])
        {
            for (int c = 0; c < 4; c++)
            {
                it->regreCoord[c] = (float)bbox[c];
            }
            it->score = (float)score[1];
            results.push_back(*it);
        }
    }
    return results;
}

vector<FaceInfo> MtcnnDetector::Onet_Detect(ncnn::Mat img, vector<FaceInfo> bboxs)
{
    vector<FaceInfo> results;

    int img_w = img.w;
    int img_h = img.h;

    for (auto it = bboxs.begin(); it != bboxs.end(); it++)
    {
        ncnn::Mat img_t;
        copy_cut_border(img, img_t, it->y[0], img_h - it->y[1], it->x[0], img_w - it->x[1]);
        ncnn::Mat in = resize(img_t, 48, 48);
        in.substract_mean_normalize(this->mean_vals, this->norm_vals);
        ncnn::Extractor ex = Onet.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score, bbox, point;
        ex.extract("prob1", score);
        ex.extract("conv6_2", bbox);
        ex.extract("conv6_3", point);
        if ((float)score[1] > threshold[2])
        {
            for (int c = 0; c < 4; c++)
            {
                it->regreCoord[c] = (float)bbox[c];
            }
            for (int p = 0; p < 5; p++)
            {
                it->landmark[2 * p] =  it->x[0] + (it->x[1] - it->x[0]) * point[p];
                it->landmark[2 * p + 1] = it->y[0] + (it->y[1] - it->y[0]) * point[p + 5];
            }
            it->score = (float)score[1];
            results.push_back(*it);
        }
    }
    return results;
}

void MtcnnDetector::Lnet_Detect(ncnn::Mat img, vector<FaceInfo> &bboxes)
{
    int img_w = img.w;
    int img_h = img.h;

    for (auto it = bboxes.begin(); it != bboxes.end(); it++)
    {
        int w = it->x[1] - it->x[0] + 1;
        int h = it->y[1] - it->y[0] + 1;
        int m = w > h ? w : h;
        m = (int)round(m * 0.25);
        if (m % 2 == 1) m++;
        m /= 2;

        ncnn::Mat in(24, 24, 15);

        for (int i = 0; i < 5; i++)
        {
            int px = it->landmark[2 * i];
            int py = it->landmark[2 * i + 1];
            ncnn::Mat cut;
            copy_cut_border(img, cut, py - m, img_h - py - m, px - m, img_w - px - m);
            ncnn::Mat resized = resize(cut, 24, 24);
            resized.substract_mean_normalize(this->mean_vals, this->norm_vals);
            for (int j = 0; j < 3; j++)
                memcpy(in.channel(3 * i + j), resized.channel(j), 24 * 24 * sizeof(float));
        }

        ncnn::Extractor ex = Lnet.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat out1, out2, out3, out4, out5;

        ex.extract("fc5_1", out1);
        ex.extract("fc5_2", out2);
        ex.extract("fc5_3", out3);
        ex.extract("fc5_4", out4);
        ex.extract("fc5_5", out5);

        if (abs(out1[0] - 0.5) > 0.35) out1[0] = 0.5f;
        if (abs(out1[1] - 0.5) > 0.35) out1[1] = 0.5f;
        if (abs(out2[0] - 0.5) > 0.35) out2[0] = 0.5f;
        if (abs(out2[1] - 0.5) > 0.35) out2[1] = 0.5f;
        if (abs(out3[0] - 0.5) > 0.35) out3[0] = 0.5f;
        if (abs(out3[1] - 0.5) > 0.35) out3[1] = 0.5f;
        if (abs(out4[0] - 0.5) > 0.35) out4[0] = 0.5f;
        if (abs(out4[1] - 0.5) > 0.35) out4[1] = 0.5f;
        if (abs(out5[0] - 0.5) > 0.35) out5[0] = 0.5f;
        if (abs(out5[1] - 0.5) > 0.35) out5[1] = 0.5f;

        it->landmark[0] += (int)round((out1[0] - 0.5) * m * 2);
        it->landmark[1] += (int)round((out1[1] - 0.5) * m * 2);
        it->landmark[2] += (int)round((out2[0] - 0.5) * m * 2);
        it->landmark[3] += (int)round((out2[1] - 0.5) * m * 2);
        it->landmark[4] += (int)round((out3[0] - 0.5) * m * 2);
        it->landmark[5] += (int)round((out3[1] - 0.5) * m * 2);
        it->landmark[6] += (int)round((out4[0] - 0.5) * m * 2);
        it->landmark[7] += (int)round((out4[1] - 0.5) * m * 2);
        it->landmark[8] += (int)round((out5[0] - 0.5) * m * 2);
        it->landmark[9] += (int)round((out5[1] - 0.5) * m * 2);
    }
}

vector<FaceInfo> MtcnnDetector::generateBbox(ncnn::Mat score, ncnn::Mat loc, float scale, float thresh)
{
    int stride = 2;
    int cellsize = 12;
    float *p = score.channel(1);
    float inv_scale = 1.0f / scale;
    vector<FaceInfo> results;
    for (int row = 0; row < score.h; row++)
    {
        for (int col = 0; col < score.w; col++)
        {
            if (*p > thresh)
            {
                FaceInfo box;
                box.score = *p;
                box.x[0] = round((stride * col + 1) * inv_scale);
                box.y[0] = round((stride * row + 1) * inv_scale);
                box.x[1] = round((stride * col + 1 + cellsize) * inv_scale);
                box.y[1] = round((stride * row + 1 + cellsize) * inv_scale);
                box.area = (box.x[1] - box.x[0]) * (box.y[1] - box.y[0]);
                int index = row * score.w + col;
                for (int c = 0; c < 4; c++)
                    box.regreCoord[c] = loc.channel(c)[index];
                results.push_back(box); 
            }
            p++;        
        }
    }
    return results;
}

bool cmpScore(FaceInfo x, FaceInfo y)
{
    if (x.score > y.score)
        return true;
    else
        return false;
}

float calcIOU(FaceInfo box1, FaceInfo box2, string mode)
{
    int maxX = max(box1.x[0], box2.x[0]);
    int maxY = max(box1.y[0], box2.y[0]);
    int minX = min(box1.x[1], box2.x[1]);
    int minY = min(box1.y[1], box2.y[1]);
    int width = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
    int height = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
    int inter = width * height;
    if (!mode.compare("union"))
        return float(inter) / (box1.area + box2.area - float(inter));
    else if (!mode.compare("min"))
        return float(inter) / (box1.area < box2.area ? box1.area : box2.area);
    else
        return 0;
}

void MtcnnDetector::doNms(vector<FaceInfo> &bboxs, float nms_thresh, string mode)
{
    if (bboxs.empty())
        return;
    sort(bboxs.begin(), bboxs.end(), cmpScore);
    for (int i = 0; i < bboxs.size(); i++)
        if (bboxs[i].score > 0)
            for (int j = i + 1; j < bboxs.size(); j++)
                if (bboxs[j].score > 0)
                {
                    float iou = calcIOU(bboxs[i], bboxs[j], mode);
                    if (iou > nms_thresh)
                        bboxs[j].score = 0;
                }
    for (auto it = bboxs.begin(); it != bboxs.end();)
        if ((*it).score == 0)
            bboxs.erase(it);
        else
            it++;
}

void MtcnnDetector::refine(vector<FaceInfo> &bboxs, int height, int width, bool flag)
{
    if (bboxs.empty())
        return;
    for (auto it = bboxs.begin(); it != bboxs.end(); it++)
    {
        float bw = it->x[1] - it->x[0] + 1;
        float bh = it->y[1] - it->y[0] + 1;
        float x0 = it->x[0] + it->regreCoord[0] * bw;
        float y0 = it->y[0] + it->regreCoord[1] * bh;
        float x1 = it->x[1] + it->regreCoord[2] * bw;
        float y1 = it->y[1] + it->regreCoord[3] * bh;

        if (flag)
        {
            float w = x1 - x0 + 1;
            float h = y1 - y0 + 1;
            float m = (h > w) ? h : w;
            x0 = x0 + w * 0.5 - m * 0.5;
            y0 = y0 + h * 0.5 - m * 0.5;
            x1 = x0 + m - 1;
            y1 = y0 + m - 1;
        }
        it->x[0] = round(x0);
        it->y[0] = round(y0);
        it->x[1] = round(x1);
        it->y[1] = round(y1);

        if (it->x[0] < 0) it->x[0] = 0;
        if (it->y[0] < 0) it->y[0] = 0;
        if (it->x[1] > width) it->x[1] = width - 1;
        if (it->y[1] > height) it->y[1] = height - 1;
        
        it->area = (it->x[1] - it->x[0]) * (it->y[1] - it->y[0]);
    }
}
