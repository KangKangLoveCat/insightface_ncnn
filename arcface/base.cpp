#include "base.h"

ncnn::Mat resize(ncnn::Mat src, int w, int h)
{
   int src_w = src.w;
   int src_h = src.h;
   unsigned char* u_src = new unsigned char[src_w * src_h * 3];
   src.to_pixels(u_src, ncnn::Mat::PIXEL_RGB);
   unsigned char* u_dst = new unsigned char[w * h * 3];
   ncnn::resize_bilinear_c3(u_src, src_w, src_h, u_dst, w, h);
   ncnn::Mat dst = ncnn::Mat::from_pixels(u_dst, ncnn::Mat::PIXEL_RGB, w, h);
   delete[] u_src;
   delete[] u_dst;
   return dst;
}

// src: source points: size=10 vector with shape (2, 5) 
// dst: dest points: size=10 vector with shape (2, 5)
// M: rotation matrix with shape (2, 3)
void getAffineMatrix(float* src_5pts, const float* dst_5pts, float* M)
{
    float src[10], dst[10];
    memcpy(src, src_5pts, sizeof(float)*10);
    memcpy(dst, dst_5pts, sizeof(float)*10);
    // initialization
    float ptmp[2];
    ptmp[0] = ptmp[1] = 0;
    for (int i = 0; i < 5; ++i) {
        ptmp[0] += src[i];
        ptmp[1] += src[5+i];
    }
    ptmp[0] /= 5;
    ptmp[1] /= 5;
    for (int i = 0; i < 5; ++i) {
        src[i] -= ptmp[0];
        src[5+i] -= ptmp[1];
        dst[i] -= ptmp[0];
        dst[5+i] -= ptmp[1];
    }
    // determine init angle
    float dst_x = (dst[3]+dst[4]-dst[0]-dst[1])/2, dst_y = (dst[8]+dst[9]-dst[5]-dst[6])/2;
    float src_x = (src[3]+src[4]-src[0]-src[1])/2, src_y = (src[8]+src[9]-src[5]-src[6])/2;
    float theta = atan2(dst_x, dst_y) - atan2(src_x, src_y);
    // determine init scale
    float scale = sqrt(pow(dst_x, 2) + pow(dst_y, 2)) / sqrt(pow(src_x, 2) + pow(src_y, 2));
    float pts1[10];
    float pts0[2];
    float _a = sin(theta), _b = cos(theta);
    pts0[0] = pts0[1] = 0;
    for (int i = 0; i < 5; ++i) {
        pts1[i] = scale*(src[i]*_b + src[i+5]*_a);
        pts1[i+5] = scale*(-src[i]*_a + src[i+5]*_b);
        pts0[0] += (dst[i] - pts1[i]);
        pts0[1] += (dst[i+5] - pts1[i+5]);
    }
    pts0[0] /= 5;
    pts0[1] /= 5;

    float sqloss = 0;
    for (int i = 0; i < 5; ++i) {
        sqloss += ((pts0[0]+pts1[i]-dst[i])*(pts0[0]+pts1[i]-dst[i])
                + (pts0[1]+pts1[i+5]-dst[i+5])*(pts0[1]+pts1[i+5]-dst[i+5]));
    }
    // optimization
    float square_sum = 0;
    for (int i = 0; i < 10; ++i) {
        square_sum += src[i]*src[i];
    }
    for (int t = 0; t < 200; ++t) {
        //cout << sqloss << endl;
        // determine angle
        _a = 0;
        _b = 0;
        for (int i = 0; i < 5; ++i) {
            _a += ((pts0[0]-dst[i])*src[i+5] - (pts0[1]-dst[i+5])*src[i]);
            _b += ((pts0[0]-dst[i])*src[i] + (pts0[1]-dst[i+5])*src[i+5]);
        }
        if (_b < 0) {
            _b = -_b;
            _a = -_a;
        }
        float _s = sqrt(_a*_a + _b*_b);
        _b /= _s;
        _a /= _s;

        for (int i = 0; i < 5; ++i) {
            pts1[i] = scale*(src[i]*_b + src[i+5]*_a);
            pts1[i+5] = scale*(-src[i]*_a + src[i+5]*_b);
        }

        // determine scale
        float _scale = 0;
        for (int i = 0; i < 5; ++i) {
            _scale += ((dst[i]-pts0[0])*pts1[i] + (dst[i+5]-pts0[1])*pts1[i+5]);
        }
        _scale /= (square_sum*scale);
        for (int i = 0; i < 10; ++i) {
            pts1[i] *= (_scale / scale);
        }
        scale = _scale;

        // determine pts0
        pts0[0] = pts0[1] = 0;
        for (int i = 0; i < 5; ++i) {
            pts0[0] += (dst[i] - pts1[i]);
            pts0[1] += (dst[i+5] - pts1[i+5]);
        }
        pts0[0] /= 5;
        pts0[1] /= 5;

        float _sqloss = 0;
        for (int i = 0; i < 5; ++i) {
            _sqloss += ((pts0[0]+pts1[i]-dst[i])*(pts0[0]+pts1[i]-dst[i])
                    + (pts0[1]+pts1[i+5]-dst[i+5])*(pts0[1]+pts1[i+5]-dst[i+5]));
        }
        if (abs(_sqloss - sqloss) < 1e-2) {
            break;
        }
        sqloss = _sqloss;
    }
    // generate affine matrix
    for (int i = 0; i < 5; ++i) {
        pts1[i] += (pts0[0] + ptmp[0]);
        pts1[i+5] += (pts0[1] + ptmp[1]);
    }
    // printMat(pts1, 2, 5);
    M[0] = _b*scale;
    M[1] = _a*scale;
    M[3] = -_a*scale;
    M[4] = _b*scale;
    M[2] = pts0[0] + ptmp[0] - scale*(ptmp[0]*_b + ptmp[1]*_a);
    M[5] = pts0[1] + ptmp[1] - scale*(-ptmp[0]*_a + ptmp[1]*_b);
}

