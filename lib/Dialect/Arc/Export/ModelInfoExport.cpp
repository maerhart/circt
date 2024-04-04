//===- ModelInfoExport.cpp - Exports model info to JSON format ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Register the MLIR translation to export model info to JSON format.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ModelInfoExport.h"
#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Support/SymCache.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/JSON.h"

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace arc;

void circt::arc::serializeModelInfoToJson(ModuleOp module,
                                          llvm::raw_ostream &os) {
  llvm::json::OStream json(os, 2);
  SymbolCache cache;
  cache.addDefinitions(module);

  json.array([&] {
    for (auto modelOp : module.getOps<ModelOp>()) {
      auto layoutTy = cast<LayoutType>(modelOp.getBody().getArgumentTypes()[0]);
      LayoutOp layoutOp =
          cast<LayoutOp>(cache.getDefinition(layoutTy.getLayoutName()));

      DenseMap<StringAttr, unsigned> offsets;
      unsigned totalSize = layoutOp.getTotalSizeAndAllOffsets(offsets);
      json.object([&] {
        json.attribute("name", modelOp.getSymName());
        json.attribute("numStateBytes", totalSize);
        json.attributeArray("states", [&] {
          for (auto entry : layoutOp.getOps<EntryOp>()) {
            if (entry.getKind() == LayoutKind::Padding)
              continue;
            json.object([&] {
              json.attribute("name", entry.getSymName());
              json.attribute("offset", offsets[entry.getSymNameAttr()]);
              // For memories this takes the width of only one element for some
              // reason.
              json.attribute(
                  "numBits",
                  arc::getBitWidth(
                      entry.getKind() == LayoutKind::Memory
                          ? entry.getType().cast<MemoryType>().getWordType()
                          : entry.getType()));
              json.attribute("type", stringifyLayoutKind(entry.getKind()));
              if (entry.getKind() == LayoutKind::Memory) {
                auto memTy = cast<MemoryType>(entry.getType());
                json.attribute("stride", memTy.getStride());
                json.attribute("depth", memTy.getNumWords());
              }
            });
          }
        });
      });
    }
  });
}

void circt::arc::registerArcModelInfoTranslation() {
  static mlir::TranslateFromMLIRRegistration modelInfoToJson(
      "export-arc-model-info", "export Arc model info in JSON format",
      [](ModuleOp module, llvm::raw_ostream &os) {
        arc::serializeModelInfoToJson(module, os);
        return success();
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<arc::ArcDialect>();
      });
}
