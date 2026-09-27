#pragma once
#include <string>

namespace onnxcc::cli {

enum class Subcommand { None, Dump };

struct DumpOptions {
    std::string model_path;
    bool        show_graph{false};
    bool        verbose{false};
};

// parse() communicates failure and help via these two flags so the
// caller (main or a test) decides the exit code — no std::exit() inside.
struct ParseResult {
    bool       error{false};      // true  → caller should exit(1)
    bool       help{false};       // true  → caller should exit(0)
    Subcommand subcommand{Subcommand::None};
    DumpOptions dump;
};

// Parses argc/argv.
//   • Errors are written to stderr; result.error is set to true.
//   • Help text is written to stdout; result.help is set to true.
//   • Never calls std::exit().
ParseResult parse(int argc, char* argv[]);

} // namespace onnxcc::cli
