#include "trt_api.h"
#include <stdexcept>


int main() {
    try 
    {
        TRTLogger logger;

        Params params;
        params.onnx_path = "E:/workspace/learn_TRT/onnx/SimpleCNN.onnx";
        params.engine_path = "E:/workspace/learn_TRT/bin/Debug/trtmodel3.engine";
        params.dynamic_Dim = true;
        params.bf16 = true;
        params.fp16 = false;
        params.int8 = false;

        auto success = createEngine(logger, params);
        
        /*
            Network definition:

            image
            |
            linear (fully connected)  input = 3, output = 2, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5]], b=[0.3, 0.8]
            |
            sigmoid
            |
            prob
        */
        // float input_data_host[] = {1, 2, 3};
        // float output_data_host[2];
        
        /*
            Network definition:

            image
                |
            conv(3x3, pad=1)  input = 1, output = 1, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5], [0.2, 0.2, 0.1]], b=0.0
                |
            relu
                |
            prob
        */
        // float input_data_host[] = {
        //     // batch 0
        //     1,   1,   1,
        //     1,   1,   1,
        //     1,   1,   1,

        //     // batch 1
        //     -1,   1,   1,
        //     1,   0,   1,
        //     1,   1,   -1
        // };
        // float output_data_host[2 * 3 * 3];
        // auto success = inference(logger, engine_path, input_data_host, output_data_host);

    } 
    catch (const std::exception& e) 
    {
        fprintf(stderr, "Error: %s\n", e.what());
        return -1;
    }

    return 0;
}
