#include "onnxcc/version.h"
#include <iostream>

int main() {
  std::cout << "Welcome to ONNXCC " << onnxcc::get_version() << " ("
            << onnxcc::get_version_codename() << ")!" << std::endl;
  return 0;
}