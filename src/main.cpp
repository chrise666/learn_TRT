#include "trt_api.h"
#include <stdexcept>


int main() {
    try 
    {
        TRTLogger logger;
        const char* onnx_path = "E:/workspace/learn_TRT/onnx/SimpleCNN.onnx";

        auto success = createEngine(logger, onnx_path);
        // inference(logger);
    } 
    catch (const std::exception& e) 
    {
        fprintf(stderr, "Error: %s\n", e.what());
        return -1;
    }

    return 0;
}
