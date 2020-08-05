//===-- Passes.td - LLHD pass definition file --------------*- tablegen -*-===//
//
// This file contains definitions for passes that work on the LLHD dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_TRANSFORMS_PASSES
#define CIRCT_DIALECT_LLHD_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def BlockArgumentToSelect : Pass<"llhd-block-argument-to-select",
                                 "llhd::ProcOp"> {
  let summary = "Convert block arguments to select operations.";
  let description = [{
    Convert block arguments to select operations by extracting the condition
    under which the block with the block argument is reached from a given
    predecessor block. Given this condition for all predecessor blocks, we
    can insert select operations to multiplex the passed values.
    This enables the Early Code Motion pass to move operations, which use
    block arguments as operands, further up in the CFG as well as the
    select operations created with this pass. It also enables the
    canonicalization pass to merge blocks which previously had block
    arguments which is necessary to completely remove all the control flow.
  }];

  let constructor = "mlir::llhd::createBlockArgumentToSelectPass()";
}

def ProcessLowering : Pass<"llhd-process-lowering", "ModuleOp"> {
  let summary = "Lowers LLHD Processes to Entities.";
  let description = [{
    TODO
  }];

  let constructor = "mlir::llhd::createProcessLoweringPass()";
}

def FunctionElimination : Pass<"llhd-function-elimination", "ModuleOp"> {
  let summary = "Deletes all functions.";
  let description = [{
    Deletes all functions in the module. In case there is still a function
    call in an entity or process, it fails.
    This pass is intended as a post-inlining pass to check if all functions
    could be successfully inlined and remove the inlined functions. This
    is necessary because Structural LLHD does not allow functions. Fails in
    the case that there is still a function call left in a `llhd.proc` or
    `llhd.entity`.
  }];

  let constructor = "mlir::llhd::createFunctionEliminationPass()";
}

def EarlyCodeMotion : Pass<"llhd-early-code-motion", "llhd::ProcOp"> {
  let summary = "Move side-effect-free instructions and llhd.prb up in the CFG";
  let description = [{
    Moves side-effect-free instructions as far up in the CFG as possible. That
    means to the earliest block where all operands are defined. Special care has
    to be given to the `llhd.prb` instruction (which is the only side-effect
    instruction moved by this pass) as it must stay in the same temporal region,
    because otherwise it might sample an older or newer state of the signal.
    This pass is designed as a preparatory pass for the Temporal Code Motion
    pass to be able to move the `llhd.drv` operations in a single TR exiting
    block without having to move operations defining the operands used by the
    `llhd.drv`. It also enables total control flow elimination as the `llhd.prb`
    instructions would not be moved by other canonicalization passes.
  }];

  let constructor = "mlir::llhd::createEarlyCodeMotionPass()";
}

#endif // CIRCT_DIALECT_LLHD_TRANSFORMS_PASSES
