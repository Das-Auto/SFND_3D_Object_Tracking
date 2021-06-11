
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

#include <chrono>

#include "helper.h"
using namespace std;

/* INIT VARIABLES AND DATA STRUCTURES */

// data location
string dataPath = "../";

// camera
string imgBasePath = dataPath + "images/";
string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
string imgFileType = ".png";
int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
int imgEndIndex = 18;  // last file index to load
int imgStepWidth = 1;
int imgFillWidth = 4; // no. of digits which make up the file index (e.g. img-0001.png)

// object detection
string yoloBasePath = dataPath + "dat/yolo/";
string yoloClassesFile = yoloBasePath + "coco.names";
string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
string yoloModelWeights = yoloBasePath + "yolov3.weights";

// Lidar
string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
string lidarFileType = ".bin";

// calibration data for camera and lidar
cv::Mat P_rect_00(3, 4, cv::DataType<double>::type); // 3x4 projection matrix after rectification
cv::Mat R_rect_00(4, 4, cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
cv::Mat RT(4, 4, cv::DataType<double>::type);        // rotation matrix and translation vector
void spinOnce(size_t imgIndex, double sensorFrameRate, int dataBufferSize, vector<DataFrame> &dataBuffer, double &ttcLidar, double &ttcCamera)
{

    bool bVis = false;
    /* LOAD IMAGE INTO BUFFER */

    // assemble filenames for current index
    ostringstream imgNumber;
    imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
    string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

    // load image from file
    cv::Mat img = cv::imread(imgFullFilename);

    // push image into data frame buffer
    DataFrame frame;
    frame.cameraImg = img;
    dataBuffer.push_back(frame);
    if (dataBuffer.size() > dataBufferSize)
    {
        dataBuffer[0] = dataBuffer[1];
        dataBuffer.pop_back();
    }
    dataBuffer.push_back(frame);

    /* DETECT & CLASSIFY OBJECTS */

    float confThreshold = 0.2;
    float nmsThreshold = 0.4;
    detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                  yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

    // cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;

    /* CROP LIDAR POINTS */

    // load 3D Lidar points from file
    string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
    std::vector<LidarPoint> lidarPoints;
    loadLidarFromFile(lidarPoints, lidarFullFilename);

    // remove Lidar points based on distance properties
    float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
    cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);

    (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

    // cout << "#3 : CROP LIDAR POINTS done" << endl;

    /* CLUSTER LIDAR POINT CLOUD */

    // associate Lidar points with camera-based ROI
    float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
    clusterLidarWithROI((dataBuffer.end() - 1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

    // Visualize 3D objects
    bVis = false;
    if (bVis)
    {
        show3DObjects((dataBuffer.end() - 1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
    }
    bVis = false;

    // REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT
    //continue; // skips directly to the next image without processing what comes beneath

    /* DETECT IMAGE KEYPOINTS */

    // convert current image to grayscale
    cv::Mat imgGray;
    cv::cvtColor((dataBuffer.end() - 1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

    // extract 2D keypoints from current image
    vector<cv::KeyPoint> keypoints; // create empty feature list for current image
    string detectorType = "FAST";

    if (detectorType.compare("SHITOMASI") == 0)
    {
        detKeypointsShiTomasi(keypoints, imgGray, false);
    }
    else
    {

        if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, false);
        }
        else
        {
            detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
    }

    // push keypoints and descriptor for current frame to end of data buffer
    (dataBuffer.end() - 1)->keypoints = keypoints;

    /* EXTRACT KEYPOINT DESCRIPTORS */

    cv::Mat descriptors;
    string descriptorType = "FREAK"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
    descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);

    // push descriptors for current frame to end of data buffer
    (dataBuffer.end() - 1)->descriptors = descriptors;

    if (dataBuffer.size() > 1) // wait until at least two images have been processed
    {

        /* MATCH KEYPOINT DESCRIPTORS */

        vector<cv::DMatch> matches;
        string matcherType = "MAT_BF";     // MAT_BF, MAT_FLANN
        string descriptorType = "DES_HOG"; // DES_BINARY, DES_HOG
        string selectorType = "SEL_NN";    // SEL_NN, SEL_KNN

        matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                         (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                         matches, descriptorType, matcherType, selectorType);

        // store matches in current data frame
        (dataBuffer.end() - 1)->kptMatches = matches;

        /* TRACK 3D OBJECT BOUNDING BOXES */

        //// STUDENT ASSIGNMENT
        //match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
        map<int, int> bbBestMatches;
        matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end() - 2), *(dataBuffer.end() - 1)); // associate bounding boxes between current and previous frame using keypoint matches

        // store matches in current data frame
        (dataBuffer.end() - 1)->bbMatches = bbBestMatches;

        /* COMPUTE TTC ON OBJECT IN FRONT */

        // loop over all BB match pairs
        for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
        {
            // find bounding boxes associates with current match
            BoundingBox *prevBB, *currBB;
            for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
            {
                if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
                {
                    currBB = &(*it2);
                }
            }

            for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
            {
                if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
                {
                    prevBB = &(*it2);
                }
            }

            // compute TTC for current match
            if (currBB->lidarPoints.size() > 0 && prevBB->lidarPoints.size() > 0) // only compute TTC if we have Lidar points
            {
                computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);

                //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)

                clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);
                computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);

            } // eof TTC computation
        }     // eof loop over all BB matches
    }
}
/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    RT.at<double>(0, 0) = 7.533745e-03;
    RT.at<double>(0, 1) = -9.999714e-01;
    RT.at<double>(0, 2) = -6.166020e-04;
    RT.at<double>(0, 3) = -4.069766e-03;
    RT.at<double>(1, 0) = 1.480249e-02;
    RT.at<double>(1, 1) = 7.280733e-04;
    RT.at<double>(1, 2) = -9.998902e-01;
    RT.at<double>(1, 3) = -7.631618e-02;
    RT.at<double>(2, 0) = 9.998621e-01;
    RT.at<double>(2, 1) = 7.523790e-03;
    RT.at<double>(2, 2) = 1.480755e-02;
    RT.at<double>(2, 3) = -2.717806e-01;
    RT.at<double>(3, 0) = 0.0;
    RT.at<double>(3, 1) = 0.0;
    RT.at<double>(3, 2) = 0.0;
    RT.at<double>(3, 3) = 1.0;

    R_rect_00.at<double>(0, 0) = 9.999239e-01;
    R_rect_00.at<double>(0, 1) = 9.837760e-03;
    R_rect_00.at<double>(0, 2) = -7.445048e-03;
    R_rect_00.at<double>(0, 3) = 0.0;
    R_rect_00.at<double>(1, 0) = -9.869795e-03;
    R_rect_00.at<double>(1, 1) = 9.999421e-01;
    R_rect_00.at<double>(1, 2) = -4.278459e-03;
    R_rect_00.at<double>(1, 3) = 0.0;
    R_rect_00.at<double>(2, 0) = 7.402527e-03;
    R_rect_00.at<double>(2, 1) = 4.351614e-03;
    R_rect_00.at<double>(2, 2) = 9.999631e-01;
    R_rect_00.at<double>(2, 3) = 0.0;
    R_rect_00.at<double>(3, 0) = 0;
    R_rect_00.at<double>(3, 1) = 0;
    R_rect_00.at<double>(3, 2) = 0;
    R_rect_00.at<double>(3, 3) = 1;

    P_rect_00.at<double>(0, 0) = 7.215377e+02;
    P_rect_00.at<double>(0, 1) = 0.000000e+00;
    P_rect_00.at<double>(0, 2) = 6.095593e+02;
    P_rect_00.at<double>(0, 3) = 0.000000e+00;
    P_rect_00.at<double>(1, 0) = 0.000000e+00;
    P_rect_00.at<double>(1, 1) = 7.215377e+02;
    P_rect_00.at<double>(1, 2) = 1.728540e+02;
    P_rect_00.at<double>(1, 3) = 0.000000e+00;
    P_rect_00.at<double>(2, 0) = 0.000000e+00;
    P_rect_00.at<double>(2, 1) = 0.000000e+00;
    P_rect_00.at<double>(2, 2) = 1.000000e+00;
    P_rect_00.at<double>(2, 3) = 0.000000e+00;

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;                       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer;                 // list of data frames which are held in memory at the same time
    bool bVis = false;                            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    // lists for all possible combinations
    // vector<string> descriptors{"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};
    //vector<string> detectors{"HARRIS", "FAST", "BRISK", "ORB", "SIFT"};

    double ttcLidar = std::numeric_limits<double>::quiet_NaN();
    double ttcCamera = std::numeric_limits<double>::quiet_NaN();
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex += imgStepWidth)
    {
        auto t_start = std::chrono::high_resolution_clock::now();

        spinOnce(imgIndex, sensorFrameRate, dataBufferSize, dataBuffer, ttcLidar, ttcCamera);
        // ttcLidar & ttcCamera are what to be sent to ROS, for first image they are NAN

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_time_sec = std::chrono::duration<double, std::milli>(t_end - t_start).count() / 1000;
        //cout << elapsed_time_sec << endl;

        cout << "Lidar : " << ttcLidar - elapsed_time_sec << " Camera : " << ttcCamera - elapsed_time_sec << endl;
    } // eof loop over all images

    return 0;
}
