//===- StructureOps.td - Process and Entity definitions ----*- tablegen -*-===//
//
// This describes the LLHD Process, Entity and control flow MLIR ops.
//
//===----------------------------------------------------------------------===//

def LLHD_EntityOp : LLHD_Op<"entity", [
    Symbol,
    FunctionLike,
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"TerminatorOp">,
    DeclareOpInterfaceMethods<CallableOpInterface>
  ]> {
  let summary = "Create an entity.";
  let description = [{
    The `llhd.entity` operation defines a new entity unit. An entity
    represents the data-flow description of how a circuit's output values
    change in reaction to changing input values.
    An entity contains one region with a single block and an implicit
    `TerminatorOp` terminator. Both the block name and terminator are
    omitted in the custom syntax. No further blocks and control-flow are
    legal inside an entity.

    Syntax:

    ```
    entity-op ::= `llhd.entity` entity-symbol `(` arg-list `)` `->`
      `(` out-list `)` attr-dict entity-region
    ```

    Example:

    ```mlir
    llhd.entity @Foo () -> () {
      %0 = llhd.const 0 : i1
      %toggle = llhd.sig %0 : i1 -> !llhd.sig<i1>
      %1 = llhd.prb %toggle : !llhd.sig<i1> -> i1
      %2 = llhd.not %1 : i1
      %dt = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
      llhd.drv %toggle, %2, %dt : !llhd.sig<i1>, i1, !llhd.time
    }
    ```
  }];

  let arguments = (ins I64Attr: $ins);
  let regions = (region SizedRegion<1>: $body);

  let verifier = [{ return ::verify(*this); }];

  let extraClassDeclaration = [{
    friend class OpTrait::FunctionLike<EntityOp>;

    // use FunctionLike traits's getBody method
    using OpTrait::FunctionLike<EntityOp>::getBody;

    /// Hooks for the input/output type enumeration in FunctionLike.
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }
    unsigned getNumFuncResults() { return getType().getNumResults(); }

    /// Hook for FunctionLike verifier.
    LogicalResult verifyType();

    /// Verifies the body of the function.
    LogicalResult verifyBody();
  }];
}

def LLHD_ProcOp : LLHD_Op<"proc", [
    Symbol,
    FunctionLike,
    IsolatedFromAbove,
    DeclareOpInterfaceMethods<CallableOpInterface>
  ]> {
  let summary = "Create a process";
  let description = [{
    A `llhd.proc` represents control-flow in a timed fashion. It allows a
    procedural description of how a circuit's output signals change in
    reaction to changing input signals. It has a region with arbitrarily
    many basic blocks. The first block is the entry block and cannot be
    targeted by the terminators. It uses `llhd.wait` as a terminator to add
    timed control-flow. Immediate control-flow with `br` or `cond_br` is
    also possible. Every process must either contain an infinite loop or
    terminate with the `llhd.halt` terminator.

    How does a process compare to functions and entities?

    | Unit     | Paradigm     | Timing    | Models                         |
    |----------|--------------|-----------|--------------------------------|
    | Function | control-flow | immediate | Computation in zero time       |
    | Process  | control-flow | timed     | Behavioral circuit description |
    | Entity   | data-flow    | timed     | Structural circuit description |

    Syntax:

    ```
    proc-op ::= `llhd.proc` proc-symbol `(` ssa-input-list `)` `->`
      `(` ssa-output-list `)` attr-dict `{` proc-region `}`
    ```

    Examples:

    ```mlir
    llhd.proc @example(%in0 : !llhd.sig<i64>, %in1 : !llhd.sig<i1>) ->
        (%out2 : !llhd.sig<i1>) {
      br ^bb1
    ^bb1:
      llhd.halt
    }
    ```
  }];

  let arguments = (ins I64Attr: $ins);
  let regions = (region AnyRegion: $body);

  let extraClassDeclaration = [{
    friend class OpTrait::FunctionLike<ProcOp>;

    /// Hooks for the input/output type enumeration in FunctionLike.
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }
    unsigned getNumFuncResults() { return getType().getNumResults(); }

    /// Hook for FunctionLike verifier.
    LogicalResult verifyType();

    /// Verifies the body of the function.
    LogicalResult verifyBody();
  }];

  let verifier = [{ return ::verify(*this); }];
}

//===----------------------------------------------------------------------===//
//=== Process and Entity Instanciation
//===----------------------------------------------------------------------===//

def LLHD_InstOp : LLHD_Op<"inst", [
    CallOpInterface,
    HasParent<"EntityOp">,
    AttrSizedOperandSegments
  ]> {
  let summary = "Instantiates a process or entity.";
  let description = [{
    Instantiates a process or entity and thus allows to build hierarchies.
    Can only be used within an entity. An instance defines a unique name
    within the entity it resides in.

    Syntax:

    ```
    inst-op ::= `llhd.inst` inst-name symbol-name `(` ssa-input-list `)` `->`
      `(` ssa-output-list `)` attr-dict `:`
      functional-type(ssa-input-list, ssa-output-list)
    ```

    Example:

    ```mlir
    llhd.inst "foo" @empty() -> () : () -> ()
    llhd.inst "bar" @proc_symbol() -> (%out0) : () -> !llhd.sig<i32>
    llhd.inst "baz" @entity_symbol(%in0, %in1) -> (%out0, %out1) :
      (!llhd.sig<i32>, !llhd.sig<i16>) -> (!llhd.sig<i8>, !llhd.sig<i4>)
    ```
  }];

  let arguments = (ins StrAttr:$name,
                       FlatSymbolRefAttr:$callee,
                       Variadic<LLHD_AnySigType>:$inputs,
                       Variadic<LLHD_AnySigType>:$outputs);

  let assemblyFormat = [{
    $name $callee `(` $inputs `)` `->` `(` $outputs `)` attr-dict `:`
    functional-type($inputs, $outputs)
  }];

  let extraClassDeclaration = [{
    StringRef getCallee() { return callee(); }
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return getAttrOfType<SymbolRefAttr>("callee");
    }
  }];

  let verifier = [{ return ::verify(*this); }];
}

//===----------------------------------------------------------------------===//
//=== Control Flow Operations
//===----------------------------------------------------------------------===//

def LLHD_TerminatorOp : LLHD_Op<"terminator", [
    Terminator,
    HasParent<"EntityOp">
  ]> {
  let summary = "Dummy terminator";
  let description = [{
    The `"llhd.terminator"` op is a dummy terminator for an `EntityOp` unit.
    It provides no further meaning other than ensuring correct termination
    of an entitiy's region. This operation provides no custom syntax and
    should never explicitly appear in LLHD's custom syntax.
  }];

  let parser = ?;
  let printer = ?;
}

def LLHD_WaitOp : LLHD_Op<"wait", [
    Terminator,
    AttrSizedOperandSegments,
    HasParent<"ProcOp">,
    DeclareOpInterfaceMethods<BranchOpInterface>
  ]> {
  let summary = "Suspends execution of a process.";
  let description = [{
    The `wait` instruction suspends execution of a process until any of the
    observed signals change or a fixed time interval has passed. Execution
    resumes at the specified basic block with the passed arguments.
    * This is a terminator instruction.
    * This instruction is only allowed in processes (`llhd.proc`).

    Example:

    ```mlir
    llhd.wait ^bb1
    llhd.wait for %time, ^bb1(%time : !llhd.time)
    llhd.wait (%0, %1 : !llhd.sig<i64>, !llhd.sig<i1>), ^bb1(%1 : !llhd.sig<i1>)
    llhd.wait for %time, (%0, %1 : !llhd.sig<i64>, !llhd.sig<i1>),
      ^bb1(%1, %0 : !llhd.sig<i1>, !llhd.sig<i64>)
    ```
  }];

  let arguments = (ins Variadic<LLHD_AnySigType>:$obs,
                       Optional<LLHD_TimeType>:$time,
                       Variadic<AnyType>:$destOps);

  let successors = (successor AnySuccessor:$dest);

  let assemblyFormat = [{
    (`for` $time^ `,`)? (`(`$obs^ `:` type($obs)`)` `,`)?
    $dest (`(` $destOps^ `:` type($destOps) `)`)? attr-dict
  }];
}

def LLHD_HaltOp : LLHD_Op<"halt", [Terminator, HasParent<"ProcOp">]> {
  let summary = "Terminates execution of a process.";
  let description = [{
    The `halt` instruction terminates execution of a process. All processes
    must halt eventually or consist of an infinite loop.

    * This is a terminator instruction
    * This instruction is only allowed in processes (`llhd.proc`).

    Syntax:

    ```
    halt-op ::= `llhd.halt`
    ```

    Example:

    ```mlir
    llhd.halt
    ```
  }];

  let assemblyFormat = "attr-dict";
}

def LLHD_ForOp : LLHD_Op<"for", [
      DeclareOpInterfaceMethods<LoopLikeOpInterface>,
      DeclareOpInterfaceMethods<RegionBranchOpInterface>,
      RecursiveSideEffects,
      AllTypesMatch<["lowerBound", "upperBound", "step"]>
  ]> {
  let summary = "For-loop operation";
  let description = [{
    The "llhd.for" operation represents a loop taking 3 SSA values as operands
    that represent the lower bound, upper bound and step respectively. The
    operation defines an SSA value for its induction variable. It has one
    region capturing the loop body. The induction variable is represented as an
    argument of this region. This SSA value always has type index, which is the
    size of the machine word. The step is a value of type index, required to be
    positive.
    The lower and upper bounds specify a half-open range: the range includes
    the lower bound but does not include the upper bound.

    The body region can contain arbitrarily many blocks, but the CGF has to be a
    DAG, so no backwards edges are allowed. At least one block has to use the
    "llhd.continue" terminator, but there can also be arbitrarily many of them.
    Calling ForOp::build will create such a region and insert
    the terminator implicitly if none is defined, so will the parsing even in
    cases when it is absent from the custom format. For example:

    ```mlir
    llhd.for %iv = %lb to %ub step %step {
      ... // body
    }
    ```

    `llhd.for` can also operate on loop-carried variables and returns the final
    values after loop termination. The initial values of the variables are
    passed as additional SSA operands to the "llhd.for" following the 3 loop
    control SSA values mentioned above (lower bound, upper bound and step). The
    operation region has equivalent arguments for each variable representing
    the value of the variable at the current iteration.

    The region must terminate with a "llhd.continue" that passes all the current
    iteration variables to the next iteration, or to the "llhd.for" result, if
    at the last iteration. Note, that when the loop-carried variables are
    present, calling ForOp::build will not insert the terminator implicitly.
    The caller must insert "llhd.continue" in that case.

    "llhd.for" results hold the final values after the last iteration.
    For example, to sum-reduce a memref:

    ```mlir
    func @reduce(%buffer: memref<1024xf32>, %lb: index,
                 %ub: index, %step: index) -> (f32) {
      // Initial sum set to 0.
      %sum_0 = constant 0.0 : f32
      // iter_args binds initial values to the loop's region arguments.
      %sum = scf.for %iv = %lb to %ub step %step
          iter_args(%sum_iter = %sum_0) -> (f32) {
        %t = load %buffer[%iv] : memref<1024xf32>
        %sum_next = addf %sum_iter, %t : f32
        // Yield current iteration sum to next iteration %sum_iter or to %sum
        // if final iteration.
        scf.yield %sum_next : f32
      }
      return %sum : f32
    }
    ```

    If the "llhd.for" defines any values, a yield must be explicitly present.
    The number and types of the "llhd.for" results must match the initial
    values in the "iter_args" binding and the yield operands.
  }];
  let arguments = (ins AnySignlessInteger:$lowerBound,
                       AnySignlessInteger:$upperBound,
                       AnySignlessInteger:$step,
                       Variadic<AnyType>:$initArgs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "Value lowerBound, Value upperBound, Value step, "
              "ValueRange iterArgs = llvm::None, "
              "function_ref<void(OpBuilder &, Location, Value, ValueRange)>"
              "    = nullptr">
  ];

  let extraClassDeclaration = [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

    Value getInductionVar() { return getLoopBody().front().getArgument(0); }
    Block::BlockArgListType getRegionIterArgs() {
      return getLoopBody().front().getArguments().drop_front();
    }
    Operation::operand_range getIterOperands() {
      return getOperands().drop_front(getNumControlOperands());
    }

    void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
    void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
    void setStep(Value step) { getOperation()->setOperand(2, step); }

    /// Number of region arguments for loop-carried values
    unsigned getNumRegionIterArgs() {
      return getLoopBody().front().getNumArguments() - 1;
    }
    /// Number of operands controlling the loop: lb, ub, step
    unsigned getNumControlOperands() { return 3; }
    /// Does the operation hold operands for loop-carried values
    bool hasIterOperands() {
      return getOperation()->getNumOperands() > getNumControlOperands();
    }
    /// Get Number of loop-carried values
    unsigned getNumIterOperands() {
      return getOperation()->getNumOperands() - getNumControlOperands();
    }

    /// Return operands used when entering the region at 'index'. These operands
    /// correspond to the loop iterator operands, i.e., those exclusing the
    /// induction variable. LoopOp only has one region, so 0 is the only valid
    /// value for `index`.
    OperandRange getSuccessorEntryOperands(unsigned index);
  }];

  let verifier = [{ return ::verify(*this); }];
}

def LLHD_YieldOp : LLHD_Op<"yield", [NoSideEffect, ReturnLike, Terminator, HasParent<"ForOp">]> {
  let summary = "loop yield and termination operation";
  let description = [{
    "scf.yield" yields an SSA value from the SCF dialect op region and
    terminates the regions. The semantics of how the values are yielded is
    defined by the parent operation.
    If "scf.yield" has any operands, the operands must match the parent
    operation's results.
    If the parent operation defines no values, then the "scf.yield" may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values are
    yielded.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result",
              [{ /* nothing to do */ }]>
  ];

  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}
