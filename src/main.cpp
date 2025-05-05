#include "build_engine.h"
#include <stdio.h>
#include <stdexcept>


int main() {
    try {
        createEngine();
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return -1;
    }

    return 0;
}
