//===- signals-runtime-wrappers.h - Simulation runtime library --*- C++ -*-===//
//
// Defines the runtime library used in LLHD simulation.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_SIMULATOR_SIGNALS_RUNTIME_WRAPPERS_H
#define CIRCT_DIALECT_LLHD_SIMULATOR_SIGNALS_RUNTIME_WRAPPERS_H

#include "State.h"

extern "C" {

//===----------------------------------------------------------------------===//
// Runtime interfaces
//===----------------------------------------------------------------------===//

/// Allocate a new signal. The index of the new signal in the state's list of
/// signals is returned.
int alloc_signal(mlir::llhd::sim::State *state, int index, char *owner,
                 uint8_t *value, int64_t size);

/// Add offset and size information for the elements of an array signal.
void add_sig_array_elements(mlir::llhd::sim::State *state, unsigned index,
                            unsigned size, unsigned numElements);

/// Add offset and size information for one element of a struct signal. Elements
/// are assumed to be added (by calling this function) in sequential order, from
/// first to last.
void add_sig_struct_element(mlir::llhd::sim::State *state, unsigned index,
                            unsigned offset, unsigned size);

/// Add allocated constructs to a process instance.
void alloc_proc(mlir::llhd::sim::State *state, char *owner,
                mlir::llhd::sim::ProcState *procState);

/// Add allocated entity state to the given instance.
void alloc_entity(mlir::llhd::sim::State *state, char *owner,
                  uint8_t *entityState);

/// Drive a value onto a signal.
void drive_signal(mlir::llhd::sim::State *state,
                  mlir::llhd::sim::SignalDetail *index, uint8_t *value,
                  uint64_t width, int time, int delta, int eps);

/// Suspend a process.
void llhd_suspend(mlir::llhd::sim::State *state,
                  mlir::llhd::sim::ProcState *procState, int time, int delta,
                  int eps);
}

#endif // CIRCT_DIALECT_LLHD_SIMULATOR_SIGNALS_RUNTIME_WRAPPERS_H
