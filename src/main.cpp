#include "build_engine.h"
#include <stdio.h>
#include <stdexcept>


int main() {
    try 
    {
        TRTLogger logger;
        createEngine(logger);
    } 
    catch (const std::exception& e) 
    {
        fprintf(stderr, "Error: %s\n", e.what());
        return -1;
    }

    return 0;
}
