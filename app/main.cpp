#include "trt_api.h"
#include "cv_api.h"
#include <stdexcept>
#include "onnx_inference.h"


// int main() {
//     try 
//     {
//         const std::string img_path = "C:/Users/ADMIN/Desktop/xxxx";
//         const std::string save_path = "C:/Users/ADMIN/Desktop/xxxx/res";

//         // 批量固定区域裁剪
//         ImageCropper::Config cfg(
//             ImageCropper::FIXED_ROI, 
//             cv::Rect(100, 50, 2048, 2048), // x,y,width,height
//             true                          // 自动调整边界
//         );

//         auto success = img_crop(img_path, save_path, cfg);
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
    const char* model_path = "E:/workspace/save/shape_classification/shape_cls.onnx";

    // 执行 ONNX 推理
    if (onnx_inference(model_path)) {
        std::cout << "ONNX 推理成功" << std::endl;
    } else {
        std::cerr << "ONNX 推理失败" << std::endl;
        return -1;
    }

    return 0;
}