//===- VectorizationInterfaces.td - IFs for Vectorization --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_VECTORIZATIONINTERFACES_TD
#define CIRCT_DIALECT_VECTORIZATIONINTERFACES_TD

include "mlir/IR/OpBase.td"

def VectorizeOpInterface : OpInterface<"VectorizeOpInterface"> {
  let description = [{
    This interface can be implemented by operations to be lowered by the
    LowerVectorization pass when inside an `arc.vectorize` operation.
  }];
  let cppNamespace = "::circt::arc";

  let methods = [
    InterfaceMethod<[{
        The default implementation is 'true' because operations implementing the
        interface typically provide unconditional implementations of the
        actual vectorization methods. However, operations with nested regions
        can override this and make it dependent on the nested operations.
      }],
      "bool", "isVectorizable",
      (ins "VectorizationKind":$kind, "mlir::SymbolTableCollection &":$symbolTable),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return true; }]>,
    InterfaceMethod<[{
      }],
      "mlir::Operation *",
      "vectorize", (ins "VectorizationKind":$kind,
                        "mlir::SymbolTableCollection &":$symbolTable,
                        "VectorizationBuilder &":$builder)>,
  ];
}

#endif // CIRCT_DIALECT_VECTORIZATIONINTERFACES_TD
