#ifndef TRT_API_H
#define TRT_API_H

#include "trt_base.h"

bool createEngine(TRTLogger& logger, const char* onnx_path=nullptr, bool dynamic_Dim=false);

bool inference(TRTLogger& logger, const char* engine_path, float input_data_host[], float output_data_host[]);

#endif