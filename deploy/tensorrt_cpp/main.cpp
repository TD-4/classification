#include <iostream>
#include <chrono>
#include "classification.h"

int main() {
    std::cout<<"CPP version: "<<__cplusplus<<std::endl;
    std::string task_type = "Classification"; // Segmentation, detection,classification
    static  ClassificationTensorRT trt;

    if (task_type == "Classification") {
        std::cout << "Classification" << std::endl;

        cv::Mat img = cv::imread("/root/cds/classification/deploy/tensorrt_cpp/oled_one/1.bmp",cv::IMREAD_GRAYSCALE);
        std::string onnxModel = "/root/cds/classification/deploy/tensorrt_cpp/b1_resnet18.onnx";
        //std::string label = "D:\\Test_TRT\\classification\\VT2_LABEL.txt";
        std::string label = "";
        std::string engine = "/root/cds/classification/deploy/tensorrt_cpp/b1_resnet18.trt";
        std::shared_ptr<ClassificationTensorRT::classifier_ctx> ctx;

        int time1, time2;
        int batch_size = 1;
        int availableGpuNums;
        trt.getDevices(&availableGpuNums);
        std::cout << "Numbers of available GPU Device is " << availableGpuNums << std::endl;

        time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        ctx = trt.classifier_initialize(onnxModel, engine, label, true, batch_size, 1, 0, 0);
        time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        std::cout << "Initialization Time : " << (time2 - time1) / 1000.0 << "s" << std::endl;


        cv::resize(img, img, cv::Size(224,224));
        img.convertTo(img, CV_32FC1, 1.0 / 255, 0);
        std::vector<float> mean_value{ 0.45734706 };
        std::vector<float> std_value{ 0.23965294 };
        img.convertTo(img, CV_32FC1, 1.0 / std_value[0], (0.0 - mean_value[0]) / std_value[0]);


        std::vector<cv::Mat>input(batch_size, img);
        std::vector<std::vector<Prediction>>batchPredictions;


        time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        batchPredictions = trt.classifier_classify(ctx, input);
        time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        std::cout << "Prediction Time : " << (time2 - time1) / 1000.0 << "s" << std::endl;


        const auto p = std::max_element(batchPredictions[0].begin(), batchPredictions[0].end(),
                                        [](const std::pair<std::basic_string<char>, float>& lhs,
                                                const std::pair<std::basic_string<char>, float>& rhs) { return lhs.second < rhs.second; });
        std::cout << "\nTOP 1 Prediction \n" << p->first << " : " << std::to_string(p->second * 100) << "%\n" << std::endl;
        std::cout << "\nTOP 5 Predictions\n";
        for (size_t i = 0; i < batchPredictions[0].size(); ++i)
        {
            Prediction p = batchPredictions[0][i];
            std::cout << p.first << " : " << std::to_string(p.second * 100) << "%\n";
        }

        std::cout << std::endl;
        time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        std::cout << "Classification Time : " << time2 - time1 << "ms" << std::endl;

    }

    return 0;
}
