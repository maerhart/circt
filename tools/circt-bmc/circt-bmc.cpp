//===- circt-bmc.cpp - The circt-bmc bounded model checker ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'circt-bmc' tool
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SMTToZ3LLVM.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/InitAllDialects.h"
#include "circt/Support/Version.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

namespace cl = llvm::cl;
using namespace circt;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Command-line options declaration
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("circt-mc Options");

static cl::opt<std::string>
    moduleName("module",
               cl::desc("Specify a named module to verify properties over."),
               cl::value_desc("module name"), cl::cat(mainCategory));

static cl::opt<int> clockBound(
    "b", cl::Required,
    cl::desc("Specify a number of clock cycles to model check up to."),
    cl::value_desc("clock cycle count"), cl::cat(mainCategory));

static cl::opt<bool>
    verifyDiagnostics("verify-diagnostics",
                      cl::desc("Check that emitted diagnostics match "
                               "expected-* lines on the corresponding line"),
                      cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool>
    splitInputFile("split-input-file",
                   cl::desc("Split the input file into pieces and process each "
                            "chunk independently"),
                   cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));

static cl::opt<std::string> inputFileName(cl::Positional, cl::Required,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

enum OutputFormat { OutputMLIR, OutputLLVM, OutputObj, OutputResult };
static cl::opt<OutputFormat> outputFormat(
    cl::desc("Specify output format"),
    cl::values(clEnumValN(OutputMLIR, "emit-mlir", "Emit LLVM MLIR dialect"),
               clEnumValN(OutputLLVM, "emit-llvm", "Emit LLVM"),
               clEnumValN(OutputObj, "emit-obj", "Emit object file"),
               clEnumValN(OutputResult, "run",
                          "Perform LEC and output result")),
    cl::init(OutputResult), cl::cat(mainCategory));

static LogicalResult checkProperty(ModuleOp input, MLIRContext &context,
                                   int bound) {
  PassManager pm(&context);
  pm.addPass(arc::createInlineModulesPass());
  LowerToBMCOptions lowerToBMCOptions;
  lowerToBMCOptions.topModule = moduleName;
  pm.addPass(createLowerToBMC(lowerToBMCOptions));
  pm.addPass(createLowerSMTToZ3LLVMPass());
  if (outputFormat == OutputLLVM)
    pm.addPass(LLVM::createDIScopeForLLVMFuncOpPass());
  if (failed(pm.run(input)))
    return failure();

  // Create the output directory or output file depending on our mode.
  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  std::string errorMessage;
  // Create an output file.
  outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
  if (!outputFile.value()) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  if (outputFormat == OutputMLIR) {
    OpPrintingFlags printingFlags;
    input->print(outputFile.value()->os(), printingFlags);
    return success();
  }

  if (outputFormat == OutputLLVM) {
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(input, llvmContext);
    if (!llvmModule)
      return failure();
    llvmModule->print(outputFile.value()->os(), nullptr);
    return success();
  }

  auto transformer = mlir::makeOptimizingTransformer(
      /*optLevel*/ 2, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

  auto handleErr = [](llvm::Error error) -> LogicalResult {
    int exitCode = EXIT_SUCCESS;
    llvm::handleAllErrors(std::move(error),
                          [&exitCode](const llvm::ErrorInfoBase &info) {
                            llvm::errs() << "Error: ";
                            info.log(llvm::errs());
                            llvm::errs() << '\n';
                            exitCode = EXIT_FAILURE;
                          });
    return failure();
  };

  auto mainFunction = dyn_cast_or_null<LLVM::LLVMFuncOp>(
      SymbolTable::lookupSymbolIn(input, "entry"));
  if (!mainFunction || mainFunction.empty())
    return handleErr(llvm::make_error<llvm::StringError>(
        "entry point not found", llvm::inconvertibleErrorCode()));

  std::optional<llvm::CodeGenOptLevel> jitCodeGenOptLevel =
      static_cast<llvm::CodeGenOptLevel>(2);
  SmallVector<StringRef, 4> sharedLibs;
  sharedLibs.push_back("/usr/lib/x86_64-linux-gnu/libz3.so");

  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.llvmModuleBuilder = nullptr;
  engineOptions.transformer = transformer;
  engineOptions.jitCodeGenOptLevel = jitCodeGenOptLevel;
  engineOptions.sharedLibPaths = sharedLibs;
  engineOptions.enableObjectDump = true;

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto expectedEngine = mlir::ExecutionEngine::create(input, engineOptions);
  if (!expectedEngine)
    return handleErr(expectedEngine.takeError());

  auto engine = std::move(*expectedEngine);

  auto expectedFPtr = engine->lookupPacked("entry");
  if (!expectedFPtr)
    return handleErr(expectedFPtr.takeError());

  if (outputFormat == OutputObj) {
    engine->dumpToObjectFile(outputFilename);
    return success();
  }

  void *empty = nullptr;
  void (*fptr)(void **) = *expectedFPtr;
  (*fptr)(&empty);

  return success();
}

static LogicalResult processBuffer(MLIRContext &context,
                                   llvm::SourceMgr &sourceMgr) {
  OwningOpRef<ModuleOp> module;
  module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module)
    return failure();

  return checkProperty(module.get(), context, clockBound);
}

/// Process a single split of the input. This allocates a source manager and
/// creates a regular or verifying diagnostic handler, depending on whether the
/// user set the verifyDiagnostics option.
static LogicalResult
processInputSplit(MLIRContext &context,
                  std::unique_ptr<llvm::MemoryBuffer> buffer) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  if (!verifyDiagnostics) {
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return processBuffer(context, sourceMgr);
  }

  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
  context.printOpOnDiagnostic(false);
  (void)processBuffer(context, sourceMgr);
  return sourceMgrHandler.verify();
}

/// Process the entire input provided by the user, splitting it up if the
/// corresponding option was specified.
static LogicalResult processInput(MLIRContext &context,
                                  std::unique_ptr<llvm::MemoryBuffer> input) {
  if (!splitInputFile)
    return processInputSplit(context, std::move(input));

  return splitAndProcessBuffer(
      std::move(input),
      [&](std::unique_ptr<llvm::MemoryBuffer> buffer, raw_ostream &) {
        return processInputSplit(context, std::move(buffer));
      },
      llvm::outs());
}

int main(int argc, char **argv) {
  // Configure the relevant command-line options.
  cl::HideUnrelatedOptions(mainCategory);
  registerMLIRContextCLOptions();

  // Parse the command-line options provided by the user.
  cl::ParseCommandLineOptions(argc, argv,
                              "circt-mc - bounded model checker\n\n"
                              "\tThis tool checks that properties hold in a "
                              "design over a symbolic bounded execution.\n");

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  // Register the supported CIRCT dialects and create a context to work with.
  DialectRegistry registry;
  registry.insert<circt::comb::CombDialect, circt::hw::HWDialect,
                  circt::seq::SeqDialect, circt::verif::VerifDialect>();
  MLIRContext context(registry);

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFileName, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    exit(false);
  }

  // Perform the logical equivalence checking; using `exit` to avoid the slow
  // teardown of the MLIR context.
  exit(failed(processInput(context, std::move(input))));
}
