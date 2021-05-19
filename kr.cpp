#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <chrono>

using namespace cv;
using namespace std;


bool horizontal(Point p, Point q)
{
    Point v(p.x - q.x, p.y - q.y);
    return v.x >= 0 && v.y >= 0 || v.x < 0 && v.y < 0;
}

Point operator-(Point p, Point q)
{
    Point w;
    w.x = p.x - q.x;
    w.y = p.y - q.y;
    return w;
}

double operator*(Point p, Point q)
{
    return p.x * q.x + p.y * q.y;
}

double len(Point p)
{
    return sqrt(p * p);
}

bool similar_lines(Point p1, Point p2, Point q1, Point q2)
{
    Point n1 = p2 - p1, n2 = q2 - q1;
    swap(n1.x, n1.y);
    swap(n2.x, n2.y);
    n1.x *= -1;
    n2.x *= -1;

    if (abs(n1 * n2) / len(n1) / len(n2) < 0.99)
        return 0;

    double C1 = -(n1 * p1);
    double C2 = -(n2 * q1);

    double x = 500;
    double Y1 = (-C1 - n1.x * x) / n1.y;
    double Y2 = (-C2 - n2.x * x) / n2.y;
    return (abs(Y1 - Y2) < 50);
}

int nom;

void analyze(Mat src, Mat& result, double lim)
{
    Mat src1, src_thres, src_thr;
   
    cvtColor(src, src1, COLOR_BGR2HSV);
    
    inRange(src1, Scalar(36, 25, 25), Scalar(70, 255, 255), src_thres);
    
    Mat dst2;
    copyTo(src, dst2, src_thres);
    imwrite("result/green_mask" + to_string(nom) + ".png", dst2);
    Mat dst, cdst, cannysrc;
    cvtColor(dst2, dst, COLOR_BGR2GRAY);
    Canny(dst, dst, 50, 120);
    dst.copyTo(cannysrc);
    cvtColor(dst, cdst, COLOR_GRAY2BGR);
    imwrite("result/canny" + to_string(nom) + ".png", cdst);
    for (int i = 0; i < src.rows; i++)
        dst.at<bool>(i, 0) = 0, dst.at<bool>(i, 1) = 0;

    for (int i = 0; i < src.cols; i++)
        dst.at<bool>(0, i) = 0, dst.at<bool>(1, i) = 0;
    
    //filter
    int sz = 15;

    int n = src.rows, m = src.cols;

    for (int i = 0; i + sz <= src.rows; i += sz)
    {
        for (int j = 0; j + sz <= src.cols; j += sz)
        {
            int ones = 0;
            int mask = 0;
            for (int k = i; k < i + sz; k++)
                for (int t = j; t < j + sz; t++)
                {
                    if (src_thres.at<bool>(k, t))
                        ones++;

                    if (dst.at<bool>(k, t))
                        mask++;
                }
            if (ones < sz * sz - 5 || mask < sz - 4 || mask > 2.5 * sz)
                for (int k = i; k < i + sz; k++)
                    for (int t = j; t < j + sz; t++)
                    {
                        src_thres.at<bool>(k, t) = 0;
                        dst.at<bool>(k, t) = 0;
                    }
        }
    }

    
    cvtColor(dst, cdst, COLOR_GRAY2BGR);
    imwrite("result/filtred" + to_string(nom) + ".png", cdst);

    vector<Vec2f> lines; // will hold the results of the detection
    HoughLines(dst, lines, 1, CV_PI / 180, 80); // runs the actual detection
    
    //clean line
    vector<Point> V, H;
    vector<pair<Point, Point>> hor, vert;
    Mat bad;
    src.copyTo(bad);
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        int gr = 0;
        
        pt1.x = cvRound(x0 + 500 * (-b));
        pt1.y = cvRound(y0 + 500 * (a));
        pt2.x = cvRound(x0 - 2000 * (-b));
        pt2.y = cvRound(y0 - 2000 * (a));
        
        if (horizontal(pt1, pt2))
        {
            H.push_back(pt2 - pt1);
            hor.push_back({ pt1, pt2 });
            line(bad, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
        }
        else
        {
            V.push_back(pt2 - pt1);
            vert.push_back({ pt1, pt2 });
            line(bad, pt1, pt2, Scalar(255, 0, 0), 2, LINE_AA);
        }
    }
    imwrite("result/all_lines" + to_string(nom) + ".png", bad);
    src.copyTo(bad);
    Point vv, hh;
    int maxv = 0, maxh = 0;
    double costhres = 0.997;
    for (int i = 0; i < V.size(); i++)
    {
        int vote = 0;
        double lenv = len(V[i]);
        for (int j = 0; j < V.size(); j++)
        {
            double k = abs(V[i] * V[j]) / lenv / len(V[j]);
            if (k > costhres)
                vote++;
        }
        if (vote > maxv)
            maxv = vote, vv = V[i];
    }
    for (int i = 0; i < H.size(); i++)
    {
        int vote = 0;
        double lenh = len(H[i]);
        for (int j = 0; j < H.size(); j++)
        {
            double k = abs(H[i] * H[j]) / lenh / len(H[j]);
            if (k > costhres)
                vote++;
        }
        if (vote > maxh)
            maxh = vote, hh = H[i];
    }
    //for (int i = 0; i < H.size(); i++)
    //{
    //    cout.precision(4); 
    //    double k = abs(H[i] * hh) / len(H[i]) / len(hh);

    //    cout << fixed << i << " " << H[i] << " " << hh << "       " << k << endl;
    //}
    double lenvv = len(vv), lenhh = len(hh);

    vector<pair<Point, Point>> trueH, trueV;
    for (int i = 0; i < hor.size(); i++)
    {
        double k = abs(H[i] * hh) / len(H[i]) / lenhh;
        if (k < costhres)
            continue;
        line(bad, hor[i].first, hor[i].second, Scalar(0, 0, 255), 2, LINE_AA);
        if (trueH.size() == 0)
            trueH.push_back(hor[i]);
        else
        {
            bool sim = 0;
            for (auto p : trueH)
                sim |= similar_lines(hor[i].first, hor[i].second, p.first, p.second);
            if (!sim)
                trueH.push_back(hor[i]);
        }
    }
    for (int i = 0; i < vert.size(); i++)
    {
        double k = abs(V[i] * vv) / len(V[i]) / lenvv;
        if (k < costhres)
            continue;
        line(bad, vert[i].first, vert[i].second, Scalar(255, 0, 0), 2, LINE_AA);
        if (trueV.size() == 0)
            trueV.push_back(vert[i]);
        else
        {
            bool sim = 0;
            for (auto p : trueV)
                sim |= similar_lines(vert[i].first, vert[i].second, p.first, p.second);
            if (!sim)
                trueV.push_back(vert[i]);
        }
    }
    imwrite("result/class_lines" + to_string(nom) + ".png", bad);

    vector<pair<Point, Point>> segments;

    auto truelines = trueH;
    for (auto p : trueV)
        truelines.push_back(p);

    for (auto p : truelines)
    {
        double k = 1.0 * (p.second.y - p.first.y) / (p.second.x - p.first.x);
        double b = p.first.y - k * p.first.x;
        vector<int> partsums(src.cols + 1, 0);
        vector<int> arr(src.cols, 0);
        vector<int> nearL(src.cols, 0), nearR(src.cols, 0);
        int last = -1;
        for (int i = 0; i < src.cols; i++)
        {
            int y = i * k + b;
            //if (y < 0 || y >= src.cols)
            //    partsums[i + 1] = partsums[i];
            //else
            //{
            //    partsums[i + 1] = partsums[i] + int(cannysrc.at<bool>(i, y) > 0);
            //    if(y + 1 < src.cols)
            //        partsums[i + 1] += int(cannysrc.at<bool>(i, y + 1) > 0);
            //}
            partsums[i + 1] = partsums[i];
            for (int j = y - 2; j <= y + 2; j++)
            {
                if (j < src.rows && j >= 0)
                {
                    arr[i] += int(cannysrc.at<bool>(j, i) > 0);
                    //src.at<Vec3b>(j, i)[0] = 255;
                }
            }
            if (arr[i] > 0)
                partsums[i + 1]++, last = i;
            nearR[i] = last;
        }
        last = arr.size();
        for (int i = arr.size() - 1; i >= 0; i--)
        {
            if (arr[i] > 0)
                last = i;
            nearL[i] = last;
        }
        int len = -1, L, R;
        for(int le = 0; le < src.cols; le++)
            for (int ri = le + 1; ri <= src.cols; ri++)
            {
                int w = partsums[ri] - partsums[le];
                if (w > lim * (ri - le) && ri - le > len)
                {
                    if (nearR[ri - 1] - nearL[le] > len)
                        L = nearL[le], R = nearR[ri - 1] + 1, len = nearR[ri - 1] - nearL[le] + 1;
                }
            }
        if (len == -1)
            continue;
        Point pt1, pt2;
        pt1.x = L, pt2.x = R - 1;
        pt1.y = int(k * L + b);
        pt2.y = int(k * R - k + b);
        segments.push_back({ pt1, pt2 });
        //break;
    }

    for (auto p : segments)
    {
        //cout << p.first << " " << p.second << endl;
        line(src, p.first, p.second, Scalar(0, 0, 255), 2, LINE_AA);
        line(cdst, p.first, p.second, Scalar(0, 0, 255), 2, LINE_AA);
        //break;
    }
    //for (auto p : trueV)
    //{
    //    line(src, p.first, p.second, Scalar(255, 0, 0), 2, LINE_AA);
    //    line(cdst, p.first, p.second, Scalar(255, 0, 0), 2, LINE_AA);
    //}
   
    //if (trueH.size() == 4)
    //{
    //    for (auto p : trueH)
    //    {
    //        Point h = p.first - p.second;
    //        cout.precision(8);
    //        cout << h << " " << hh << "   " << fixed << abs(h * hh) / lenhh / len(h) << endl;
    //    }
    //    cout << "\n\n\n\n\n";
    //}

    src.copyTo(result);
}

void analyze_easy(Mat src, Mat& result)
{
    Mat src1, src_thres, src_thr;

    cvtColor(src, src1, COLOR_BGR2HSV);

    //inRange(src1, Scalar(36, 25, 25), Scalar(70, 255, 255), src_thres);

    Mat dst2;
    //copyTo(src, dst2, src_thres);
    src.copyTo(dst2);
    Mat dst;

    Canny(dst2, dst, 50, 120, 3, true);

    vector<Vec2f> lines; // will hold the results of the detection
    HoughLines(dst, lines, 1, CV_PI / 180, 80, 0, 0); // runs the actual detection

    //clean line
    vector<Point> V, H;
    vector<pair<Point, Point>> hor, vert;
    for (size_t i = 0; i < min<int>(lines.size(), 13); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        int gr = 0;

        pt1.x = cvRound(x0 + 100 * (-b));
        pt1.y = cvRound(y0 + 100 * (a));
        pt2.x = cvRound(x0 - 2000 * (-b));
        pt2.y = cvRound(y0 - 2000 * (a));

        if (horizontal(pt1, pt2))
        {
            H.push_back(pt2 - pt1);
            hor.push_back({ pt1, pt2 });
            line(src, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
        }
        else
        {
            V.push_back(pt2 - pt1);
            vert.push_back({ pt1, pt2 });
            line(src, pt1, pt2, Scalar(255, 0, 0), 2, LINE_AA);
        }
    }
    src.copyTo(result);
}





pair<double, double> F_metric(Mat& src, Mat& best, Mat& res, bool print = false)
{

    Mat msk1(src.rows, src.cols, CV_8U), msk2(src.rows, src.cols, CV_8U);
    int TP = 0, TN = 0, FP = 0, FN = 0;
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++)
        {
            bool f1 = best.at<Vec3b>(i, j)[0] != src.at<Vec3b>(i, j)[0] ||
                best.at<Vec3b>(i, j)[1] != src.at<Vec3b>(i, j)[1] || best.at<Vec3b>(i, j)[2] != src.at<Vec3b>(i, j)[2];
            bool f2 = res.at<Vec3b>(i, j)[0] != src.at<Vec3b>(i, j)[0] ||
                res.at<Vec3b>(i, j)[1] != src.at<Vec3b>(i, j)[1] || res.at<Vec3b>(i, j)[2] != src.at<Vec3b>(i, j)[2];
            msk1.at<char>(i, j) = f1 * 255;
            msk2.at<char>(i, j) = f2 * 255;
            if (f1 && f2)
                TP++;
            if (!f1 && !f2)
                TN++;
            if (f1 && !f2)
                FN++;
            if (!f1 && f2)
                FP++;
        }
    Mat diff;
    bitwise_and(msk1, msk2, diff);
    if (print)
    {
        imwrite("result/diff_orig"+ to_string(nom) + ".png", msk1);
        imwrite("result/diff_real" + to_string(nom) + ".png", msk2);
        imwrite("result/diff" + to_string(nom) + ".png", diff);
    }
    return { 1.0 * TP / (TP + FP), 1.0 * TP / (TP + FN) };
}

int main() {
    double pr1 = 0, r1 = 0, f1 = 0;
    cout.precision(5);

    for (double qw = 0.7; qw <= 0.95; qw += 0.01)
    {
        pr1 = 0, r1 = 0, f1 = 0;
        for (int i = 0; i < 16; i++) {
            nom = i;
            Mat src, orig, res, best, bad;
            src = imread("photo/ref" + to_string(i) + ".png");
            best = imread("ideal/ref" + to_string(i) + ".png");

            src.copyTo(orig);
            double prec1, rec1;
            analyze(orig, res, qw);
            tie(prec1, rec1) = F_metric(src, best, res, true);
            src.copyTo(orig);
            imwrite("result/res" + to_string(i) + ".png", res);
            pr1 += prec1;
            r1 += rec1;
            f1 += 2 * prec1 * rec1 / (prec1 + rec1);
            //cout << fixed << prec1 << " " << rec1 << " " << 2 * prec1 * rec1 / (prec1 + rec1) << "\n";
            //cout << "\n";
        }
        cout << qw << " F:" << f1 / 16 << "\n";
        /*pr1 /= 7, r1 /= 7;
        cout << qw << " Average Precision #1: " << pr1 << "\n";
        cout << qw << " Average Recall #1: " << r1 << "\n\n";*/
    }

    
        /*pr1 /= 7, r1 /= 7;
        cout << qw << " Average Precision #1: " << pr1 << "\n";
        cout << qw << " Average Recall #1: " << r1 << "\n\n";*/
    }
}
