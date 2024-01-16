//===- circt-lec.cpp - The circt-lec driver ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file initiliazes the 'circt-lec' tool, which interfaces with a logical
/// engine to allow its user to check whether two input circuit descriptions
/// are equivalent, and when not provides a counterexample as for why.
///
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SMTToZ3LLVM.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/InitAllDialects.h"
#include "circt/Support/Version.h"
#include "circt/Tools/circt-lec/Passes.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

namespace cl = llvm::cl;

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Command-line options declaration
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("circt-lec Options");

static cl::opt<std::string> moduleName1(
    "c1", cl::Required,
    cl::desc("Specify a named module for the first circuit of the comparison"),
    cl::value_desc("module name"), cl::cat(mainCategory));

static cl::opt<std::string> moduleName2(
    "c2", cl::Required,
    cl::desc("Specify a named module for the second circuit of the comparison"),
    cl::value_desc("module name"), cl::cat(mainCategory));

static cl::opt<std::string> fileName1(cl::Positional, cl::Required,
                                      cl::desc("<input file>"),
                                      cl::cat(mainCategory));

static cl::opt<std::string> fileName2(cl::Positional, cl::desc("[input file]"),
                                      cl::cat(mainCategory));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
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

static cl::opt<bool>
    verbose("v", cl::init(false),
            cl::desc("Print extensive execution progress information"),
            cl::cat(mainCategory));

// The following options are stored externally for their value to be accessible
// to other components of the tool.
bool statisticsOpt;
static cl::opt<bool, true> statistics(
    "s", cl::location(statisticsOpt), cl::init(false),
    cl::desc("Print statistics about the logical engine's execution"),
    cl::cat(mainCategory));

//===----------------------------------------------------------------------===//
// Tool implementation
//===----------------------------------------------------------------------===//

/// This functions initializes the various components of the tool and
/// orchestrates the work to be done. It first parses the input files, then it
/// traverses their IR to export the logical constraints from the given circuit
/// description to an internal circuit representation, lastly, these will be
/// compared and solved for equivalence.
static LogicalResult executeLEC(MLIRContext &context) {
  // Parse the provided input files.
  OwningOpRef<ModuleOp> file1 = parseSourceFile<ModuleOp>(fileName1, &context);
  if (!file1)
    return failure();

  OwningOpRef<ModuleOp> file2;
  if (!fileName2.empty()) {
    file2 = parseSourceFile<ModuleOp>(fileName2, &context);
    if (!file2)
      return failure();
  }

  // If two files are specified, copy the contents of the builtin.module of the
  // second file into the builtin.module of the first file.
  if (!fileName2.empty()) {
    IRRewriter rewriter(&context);
    rewriter.setInsertionPointToEnd(file1.get().getBody());
    rewriter.inlineBlockBefore(file2.get().getBody(), file1.get().getBody(),
                               file1.get().getBody()->end());
  }

  // Create the output directory or output file depending on our mode.
  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  std::string errorMessage;
  // Create an output file.
  outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
  if (!outputFile.value()) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  PassManager pm(&context);
  pm.addPass(arc::createInlineModulesPass());
  LowerToLECOptions lowerToLECOptions;
  lowerToLECOptions.firstModule = moduleName1;
  lowerToLECOptions.secondModule = moduleName2;
  pm.addPass(createLowerToLEC(lowerToLECOptions));
  pm.addPass(createLowerSMTToZ3LLVMPass());
  if (outputFormat == OutputLLVM)
    pm.addPass(LLVM::createDIScopeForLLVMFuncOpPass());
  if (failed(pm.run(file1.get())))
    return failure();

  if (outputFormat == OutputMLIR) {
    OpPrintingFlags printingFlags;
    file1->print(outputFile.value()->os(), printingFlags);
    return success();
  }

  if (outputFormat == OutputLLVM) {
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(file1.get(), llvmContext);
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
      SymbolTable::lookupSymbolIn(file1.get(), "entry"));
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

  auto expectedEngine =
      mlir::ExecutionEngine::create(file1.get(), engineOptions);
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

/// The entry point for the `circt-lec` tool:
/// configures and parses the command-line options,
/// registers all dialects within a MLIR context,
/// and calls the `executeLEC` function to do the actual work.
int main(int argc, char **argv) {
  // Configure the relevant command-line options.
  cl::HideUnrelatedOptions(mainCategory);
  registerMLIRContextCLOptions();
  cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << circt::getCirctVersion() << '\n'; });

  // Parse the command-line options provided by the user.
  cl::ParseCommandLineOptions(
      argc, argv,
      "circt-lec - logical equivalence checker\n\n"
      "\tThis tool compares two input circuit descriptions to determine whether"
      " they are logically equivalent.\n");

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  // Register the supported CIRCT dialects and create a context to work with.
  DialectRegistry registry;
  registry.insert<circt::comb::CombDialect, circt::hw::HWDialect,
                  circt::smt::SMTDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::BuiltinDialect>();
  mlir::func::registerInlinerExtension(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);

  // Setup of diagnostic handling.
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  // Avoid printing a superfluous note on diagnostic emission.
  context.printOpOnDiagnostic(false);

  // Perform the logical equivalence checking; using `exit` to avoid the slow
  // teardown of the MLIR context.
  exit(failed(executeLEC(context)));
}
