#ifndef TRT_API_H
#define TRT_API_H

#include "trt_base.h"

bool createEngine(TRTLogger& logger, const char* onnx_path=nullptr, bool dynamic_Dim=false);

void inference(TRTLogger logger);

#endif