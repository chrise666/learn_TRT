// #include "trt_api.h"
#include "cv_api.h"
#include <stdexcept>
#include "onnx_inference.h"


// int main() {
//     try 
//     {
//         const std::string img_path = "C:/Users/ADMIN/Desktop/xxxx";
//         const std::string save_path = "C:/Users/ADMIN/Desktop/xxxx/res";
//         auto success = img_crop(img_path);
//     } 
//     catch (const std::exception& e) 
//     {
//         fprintf(stderr, "Error: %s\n", e.what());
//         return -1;
//     }

//     return 0;
// }

// int main() {
//     try 
//     {
//         TRTLogger logger;

//         Params params;
//         params.onnx_path = "E:/workspace/learn_TRT/save/onnx/SimpleCNN.onnx";
//         params.engine_path = "E:/workspace/learn_TRT/save/engine/trtmodel.engine";
//         params.dynamic_Dim = true;
//         params.bf16 = true;
//         params.fp16 = false;
//         params.int8 = false;

//         auto success = createEngine(logger, params);
//     } 
//     catch (const std::exception& e) 
//     {
//         fprintf(stderr, "Error: %s\n", e.what());
//         return -1;
//     }

//     return 0;
// }

// int main() {
//     try 
//     {       
//         /*
//             Network definition:

//             image
//             |
//             linear (fully connected)  input = 3, output = 2, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5]], b=[0.3, 0.8]
//             |
//             sigmoid
//             |
//             prob
//         */
//         float input_data_host[] = {1, 2, 3};
//         float output_data_host[2];
        
//         /*
//             Network definition:

//             image
//                 |
//             conv(3x3, pad=1)  input = 1, output = 1, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5], [0.2, 0.2, 0.1]], b=0.0
//                 |
//             relu
//                 |
//             prob
//         */
//         float input_data_host[] = {
//             // batch 0
//             1,   1,   1,
//             1,   1,   1,
//             1,   1,   1,

//             // batch 1
//             -1,   1,   1,
//             1,   0,   1,
//             1,   1,   -1
//         };
//         float output_data_host[2 * 3 * 3];
//         // auto success = inference(logger, engine_path, input_data_host, output_data_host);
//     } 
//     catch (const std::exception& e) 
//     {
//         fprintf(stderr, "Error: %s\n", e.what());
//         return -1;
//     }

//     return 0;
// }

// int main() {
//     try {
//         const std::string img_path = "C:/Users/ADMIN/Desktop/xxxx";
//         const std::string save_path = "C:/Users/ADMIN/Desktop/xxxx/res";
//         cv::Size target_size(2048, 2048);

//         auto success = img_resize(img_path, save_path, target_size);
//     }
//     catch (const std::exception& e) {
//         std::cerr << "错误: " << e.what() << std::endl;
//         return EXIT_FAILURE;
//     }
//     return EXIT_SUCCESS;
// }

int main() {
    const char* model_path = "E:/workspace/learn_AI/save/shape_classification/resnet50.onnx";
    std::string img_path = "E:/workspace/learn_AI/save/shape_classification/img";

    const std::vector<std::string> CLASS_NAMES = {"点状", "团状", "线状"};

    // 预处理图像
    int target_height = 128;
    int target_width = 128;

    try {
        auto images = preprocess_image(img_path, target_height, target_width);

        // 执行 ONNX 推理
        for (auto &img : images) {    
            auto result = onnx_inference(model_path, img.image.ptr<float>(), img.image.rows, img.image.cols);

            // 显示结果
            std::cout << "\n图片" << img.filename << "推理结果:" << std::endl;
            std::cout << "  推理时间: " << result.inference_time << " ms" << std::endl;
            std::cout << "  最高概率类别: " << CLASS_NAMES[result.top_class_id] 
                        << " (" << result.top_probability * 100 << "%)" << std::endl;

            std::cout << "\n所有类别概率:" << std::endl;
            for (size_t i = 0; i < result.probabilities.size(); ++i) {
                std::cout << "  类别 " << CLASS_NAMES[i] << ": " 
                            << std::fixed << std::setprecision(2) 
                            << result.probabilities[i] * 100 << "%" << std::endl;
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}