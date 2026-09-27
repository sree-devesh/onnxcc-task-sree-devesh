#include <gtest/gtest.h>
#include "onnxcc/cli/cli.hpp"

// Build a mutable argv array from an initializer list of string literals.
// Each Argv instance owns its own storage so tests never share state.
struct Argv {
    std::vector<std::string> storage;
    std::vector<char*>       ptrs;

    explicit Argv(std::initializer_list<const char*> args) {
        storage.reserve(args.size());
        for (const char* a : args) {
            storage.emplace_back(a);
        }
        ptrs.reserve(storage.size());
        for (auto& s : storage) {
            ptrs.push_back(s.data());
        }
    }

    int    argc()       { return static_cast<int>(ptrs.size()); }
    char** argv()       { return ptrs.data(); }
};

// ── dump subcommand ──────────────────────────────────────────────────────────

// dump without --model must fail with error
TEST(CliDump, RequiresModel) {
    Argv args{"onnxcc", "dump"};
    auto r = onnxcc::cli::parse(args.argc(), args.argv());
    EXPECT_TRUE(r.error);
    EXPECT_FALSE(r.help);
}

// dump --model <path> must succeed and store the path
TEST(CliDump, ParsesModelPath) {
    Argv args{"onnxcc", "dump", "--model", "net.onnx"};
    auto r = onnxcc::cli::parse(args.argc(), args.argv());
    EXPECT_FALSE(r.error);
    EXPECT_FALSE(r.help);
    EXPECT_EQ(r.subcommand, onnxcc::cli::Subcommand::Dump);
    EXPECT_EQ(r.dump.model_path, "net.onnx");
    EXPECT_FALSE(r.dump.show_graph);
    EXPECT_FALSE(r.dump.verbose);
}

// --show-graph and --verbose must be captured when present
TEST(CliDump, ParsesOptionalFlags) {
    Argv args{"onnxcc", "dump", "--model", "m.onnx", "--show-graph", "--verbose"};
    auto r = onnxcc::cli::parse(args.argc(), args.argv());
    EXPECT_FALSE(r.error);
    EXPECT_TRUE(r.dump.show_graph);
    EXPECT_TRUE(r.dump.verbose);
}

// dump --help must set help=true and not error
TEST(CliDump, HelpFlagSucceeds) {
    Argv args{"onnxcc", "dump", "--help"};
    auto r = onnxcc::cli::parse(args.argc(), args.argv());
    EXPECT_FALSE(r.error);
    EXPECT_TRUE(r.help);
}

// dump -h is the short form of --help
TEST(CliDump, ShortHelpFlagSucceeds) {
    Argv args{"onnxcc", "dump", "-h"};
    auto r = onnxcc::cli::parse(args.argc(), args.argv());
    EXPECT_FALSE(r.error);
    EXPECT_TRUE(r.help);
}

// ── top-level ────────────────────────────────────────────────────────────────

// no arguments at all must error
TEST(CliTopLevel, NoArgsIsError) {
    Argv args{"onnxcc"};
    auto r = onnxcc::cli::parse(args.argc(), args.argv());
    EXPECT_TRUE(r.error);
}

// --help must succeed (exit 0 in real use)
TEST(CliTopLevel, LongHelpSucceeds) {
    Argv args{"onnxcc", "--help"};
    auto r = onnxcc::cli::parse(args.argc(), args.argv());
    EXPECT_FALSE(r.error);
    EXPECT_TRUE(r.help);
}

// -h is the short form of --help
TEST(CliTopLevel, ShortHelpSucceeds) {
    Argv args{"onnxcc", "-h"};
    auto r = onnxcc::cli::parse(args.argc(), args.argv());
    EXPECT_FALSE(r.error);
    EXPECT_TRUE(r.help);
}

// an unrecognised subcommand must error
TEST(CliTopLevel, UnknownSubcommandIsError) {
    Argv args{"onnxcc", "run"};
    auto r = onnxcc::cli::parse(args.argc(), args.argv());
    EXPECT_TRUE(r.error);
}
