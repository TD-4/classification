//
// Created by fusimeng on 2021/5/6.
//

#ifndef TENSORRT_CDS_CLASSIFICATION_H
#define TENSORRT_CDS_CLASSIFICATION_H

// include Opencv
#include <utility>
#include <memory>
#include <vector>
#include <stddef.h>

#include "opencv4/opencv2/cudaarithm.hpp"
#include "opencv4/opencv2/cudaimgproc.hpp"
#include "opencv4/opencv2/cudawarping.hpp"
#include "opencv4/opencv2/highgui/highgui.hpp"

#ifdef __cplusplus
extern "C" {
#endif
    //---------------------------------分类、检测、分割结果表示---------------------------------
    //@brief vector<Prediction>可以用来表示一张图的分类结果，Prediction表示<类别，置信度>
    typedef std::pair<std::string, float> Prediction;

    //@brief cv::Mat           用来表示单张图分割任务结果，舍弃了置信度信息

    /**
     * @brief vector<Detection>可以用来表示一张图的检测结果，bbox表示检测框相对格点偏移量
     * @param bbox[4]			检测框相对格点偏移量
     * @param det_confidence]  检测框中包含目标的置信度
     * @param class_id			检测框对应目标类别
     * @param class_confidence	检测框目标类别为第class_id类的置信度
    */
    typedef struct {
        float bbox[4];
        float det_confidence;
        float class_id;
        float class_confidence;
    } Detection;

    //---------------------------------网络类型---------------------------------
    enum class NetWorkType : int
    {
        kLUSC = 0, //!<  自研分类网络
        kLUSS = 1,  //!< 自研分割网络
        kLUSD = 2,  //!< 自研检测网络
        kRES18 = 3, //!<  ResNet18
        kLED = 4   //!<   LEDNet
    };

    // ---------------------------------API 定义---------------------------------
    class ClassificationTensorRT {
public:
    typedef struct classifier_ctx classifier_ctx;


    //@brief 获取可用的GPU数量
    void getDevices(int* num);

    int device_;
    //@brief 设置当前使用的GPU
    void setDevice();

    /**
     * @brief 使用pytorch等框架训练得到的模型生成TRT引擎，需要首先转成onnx格式

     * @param onnx_file	 模型配置文件和参数文件,.onnx格式
     * @param label_file	 标签信息,.txt格式即可，不指定则按类别序号生成
     * @param engine_file	 生成TRT引擎文件名字,.engine后缀，如果存在该名字文件，则直接解析引擎文件，不然，则从前两个文件中生成
     * @param attachSoftmax 是否在模型后附加一个softmax层，分类和分割模型都有一个softmax层，但siamese网络没有
     * @param maxBatch		 指定最大的batchsize，小于该数值的一批图也可以执行前向推断，使用时batch最好等于该数值，性能会有所提升
     * @param kContextsPerDevice	生成多个context供多个线程同时调用，默认1
     * @param numGpus		 指定使用第numGpus块GPU，编号从0开始
     * @param data_type	 定义推断的精度，TRT支持半精度，int8等多种精度，该接口中只支持完整精度和半精度，对应0，1，默认为0
    */
    std::shared_ptr<classifier_ctx> classifier_initialize(
            std::string& onnx_file,
            std::string& engine_file,
            std::string& label_file,
            bool attachSoftmax,
            int maxBatch,
            int kContextsPerDevice,
            int numGpus,
            int data_type
    );


    /**
     * @brief          执行分类的函数，每一批图调用一次

     * @param ctx		初始化生成具有前向功能的context池后，执行推断得到分类结果
     * @param batchImg	用作推断的一批图片，小心这里的图片数量不可超过初始化时用到的maxBatch
     see classifier_initialize()
    */
    std::vector<std::vector<Prediction>> classifier_classify(
            std::shared_ptr<classifier_ctx>ctx,
            std::vector<cv::Mat>& batchImg);


    /**
     * @brief          执行分割的函数，每一批图调用一次

    * @param ctx		初始化生成具有前向功能的context池后，执行推断得到分类结果
    * @param batchImg	用作推断的一批图片，小心这里的图片数量不可超过初始化时用到的maxBatch
    see classifier_initialize()
    */
    std::vector<cv::Mat> classifier_segment(
            std::shared_ptr<classifier_ctx> ctx,
            std::vector<cv::Mat>& batchImg,
            bool probFormat = false);



    /**
     * @brief          执行检测的函数，每一批图调用一次

     * @param ctx		初始化生成具有前向功能的context池后，执行推断得到分类结果
     * @param batchImg	用作推断的一批图片，小心这里的图片数量不可超过初始化时用到的maxBatch
        see classifier_initialize()
    */
    //std::vector<std::vector<Detection>> classifier_detect(classifier_ctx* ctx, std::vector<cv::Mat>& batchImg);


    /**
     * @brief          执行相似度比对的函数，每一批图调用一次

     * @param ctx		初始化生成具有前向功能的context池后，执行推断得到相似度比对结果
     * @param pairImg	用作推断的成对图片，只支持两张图片，图片的先后顺序是无关的
     * @param threshold差异度阈值，差异度小于该数值的成对图像会被判定为相似（true）
        see classifier_initialize()
    */
    bool classifier_judge(std::shared_ptr<classifier_ctx>ctx, std::vector<cv::Mat>& pairImg, float threshold);

    //@brief 释放资源
    //void classifier_destroy(classifier_ctx* ctx);

};
#ifdef __cplusplus
}
#endif
#endif //TENSORRT_CDS_CLASSIFICATION_H
