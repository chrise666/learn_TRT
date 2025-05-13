#include "build_engine.h"
#include "inference.h"
#include <stdio.h>
#include <stdexcept>


int main() {
    try 
    {
        TRTLogger logger;
        // auto success = createEngine(logger);
        inference(logger);
    } 
    catch (const std::exception& e) 
    {
        fprintf(stderr, "Error: %s\n", e.what());
        return -1;
    }

    return 0;
}
