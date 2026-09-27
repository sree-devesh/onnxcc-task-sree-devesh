#include "onnxcc/cli/cli.hpp"
#include "onnxcc/version.h"
#include <iostream>

int main(int argc, char* argv[]) {
    auto result = onnxcc::cli::parse(argc, argv);

    if (result.error) {
        return 1;
    }
    if (result.help) {
        return 0;
    }

    if (result.subcommand == onnxcc::cli::Subcommand::Dump) {
        if (result.dump.verbose) {
            std::cout << "onnxcc " << onnxcc::get_version()
                      << " (" << onnxcc::get_version_codename() << ")\n";
        }
        std::cout << "model: " << result.dump.model_path << "\n";
        if (result.dump.show_graph) {
            std::cout << "(graph display not yet implemented)\n";
        }
    }

    return 0;
}
