#include "onnxcc/cli/cli.hpp"
#include "onnxcc/third_party/cxxopts.hpp"
#include <iostream>

namespace onnxcc::cli {

// Prints top-level usage to the given stream.
static void print_top_level_help(std::ostream& out) {
    out << "Usage: onnxcc <subcommand> [options]\n"
           "\n"
           "Subcommands:\n"
           "  dump    Print information about an ONNX model\n"
           "\n"
           "Run 'onnxcc <subcommand> --help' for subcommand-specific help.\n";
}

ParseResult parse(int argc, char* argv[]) {
    ParseResult result;

    // ── no arguments at all ────────────────────────────────────────────────
    if (argc < 2) {
        std::cerr << "error: no subcommand given\n\n";
        print_top_level_help(std::cerr);
        result.error = true;
        return result;
    }

    const std::string sub{argv[1]};

    // ── top-level --help / -h ──────────────────────────────────────────────
    if (sub == "--help" || sub == "-h") {
        print_top_level_help(std::cout);
        result.help = true;
        return result;
    }

    // ── dump subcommand ────────────────────────────────────────────────────
    if (sub == "dump") {
        result.subcommand = Subcommand::Dump;

        cxxopts::Options opts("onnxcc dump",
                              "Print information about an ONNX model");
        opts.add_options()
            ("model",
             "Path to the .onnx model file",
             cxxopts::value<std::string>())
            ("show-graph",
             "Print the computation graph",
             cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("verbose",
             "Enable verbose output",
             cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("h,help", "Show this help message");

        try {
            // Pass (argc-1, argv+1) so cxxopts treats "dump" as the program
            // name and real options start at argv[2].
            auto parsed = opts.parse(argc - 1, argv + 1);

            if (parsed.count("help")) {
                std::cout << opts.help() << "\n";
                result.help = true;
                return result;
            }

            if (!parsed.count("model")) {
                std::cerr << "error: dump requires --model <path>\n\n"
                          << opts.help() << "\n";
                result.error = true;
                return result;
            }

            result.dump.model_path = parsed["model"].as<std::string>();
            result.dump.show_graph = parsed["show-graph"].as<bool>();
            result.dump.verbose    = parsed["verbose"].as<bool>();

        } catch (const cxxopts::exceptions::parsing& e) {
            std::cerr << "error: " << e.what() << "\n";
            result.error = true;
        }

        return result;
    }

    // ── unknown subcommand ─────────────────────────────────────────────────
    std::cerr << "error: unknown subcommand '" << sub << "'\n\n";
    print_top_level_help(std::cerr);
    result.error = true;
    return result;
}

} // namespace onnxcc::cli
