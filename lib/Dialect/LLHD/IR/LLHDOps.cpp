//===- LLHDOps.cpp - Implement the LLHD operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the LLHD ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/HW/CustomDirectiveImpl.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/ModuleImplementation.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace mlir;
using namespace llhd;

template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT = function_ref<ElementValueT(ElementValueT)>>
static Attribute constFoldUnaryOp(ArrayRef<Attribute> operands,
                                  const CalculationT &calculate) {
  assert(operands.size() == 1 && "unary op takes one operand");
  if (!operands[0])
    return {};

  if (auto val = dyn_cast<AttrElementT>(operands[0])) {
    return AttrElementT::get(val.getType(), calculate(val.getValue()));
  } else if (auto val = dyn_cast<SplatElementsAttr>(operands[0])) {
    // Operand is a splat so we can avoid expanding the value out and
    // just fold based on the splat value.
    auto elementResult = calculate(val.getSplatValue<ElementValueT>());
    return DenseElementsAttr::get(val.getType(), elementResult);
  }
  if (auto val = dyn_cast<ElementsAttr>(operands[0])) {
    // Operand is ElementsAttr-derived; perform an element-wise fold by
    // expanding the values.
    auto valIt = val.getValues<ElementValueT>().begin();
    SmallVector<ElementValueT, 4> elementResults;
    elementResults.reserve(val.getNumElements());
    for (size_t i = 0, e = val.getNumElements(); i < e; ++i, ++valIt)
      elementResults.push_back(calculate(*valIt));
    return DenseElementsAttr::get(val.getType(), elementResults);
  }
  return {};
}

template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT = function_ref<
              ElementValueT(ElementValueT, ElementValueT, ElementValueT)>>
static Attribute constFoldTernaryOp(ArrayRef<Attribute> operands,
                                    const CalculationT &calculate) {
  assert(operands.size() == 3 && "ternary op takes three operands");
  if (!operands[0] || !operands[1] || !operands[2])
    return {};

  if (isa<AttrElementT>(operands[0]) && isa<AttrElementT>(operands[1]) &&
      isa<AttrElementT>(operands[2])) {
    auto fst = cast<AttrElementT>(operands[0]);
    auto snd = cast<AttrElementT>(operands[1]);
    auto trd = cast<AttrElementT>(operands[2]);

    return AttrElementT::get(
        fst.getType(),
        calculate(fst.getValue(), snd.getValue(), trd.getValue()));
  }
  if (isa<SplatElementsAttr>(operands[0]) &&
      isa<SplatElementsAttr>(operands[1]) &&
      isa<SplatElementsAttr>(operands[2])) {
    // Operands are splats so we can avoid expanding the values out and
    // just fold based on the splat value.
    auto fst = cast<SplatElementsAttr>(operands[0]);
    auto snd = cast<SplatElementsAttr>(operands[1]);
    auto trd = cast<SplatElementsAttr>(operands[2]);

    auto elementResult = calculate(fst.getSplatValue<ElementValueT>(),
                                   snd.getSplatValue<ElementValueT>(),
                                   trd.getSplatValue<ElementValueT>());
    return DenseElementsAttr::get(fst.getType(), elementResult);
  }
  if (isa<ElementsAttr>(operands[0]) && isa<ElementsAttr>(operands[1]) &&
      isa<ElementsAttr>(operands[2])) {
    // Operands are ElementsAttr-derived; perform an element-wise fold by
    // expanding the values.
    auto fst = cast<ElementsAttr>(operands[0]);
    auto snd = cast<ElementsAttr>(operands[1]);
    auto trd = cast<ElementsAttr>(operands[2]);

    auto fstIt = fst.getValues<ElementValueT>().begin();
    auto sndIt = snd.getValues<ElementValueT>().begin();
    auto trdIt = trd.getValues<ElementValueT>().begin();
    SmallVector<ElementValueT, 4> elementResults;
    elementResults.reserve(fst.getNumElements());
    for (size_t i = 0, e = fst.getNumElements(); i < e;
         ++i, ++fstIt, ++sndIt, ++trdIt)
      elementResults.push_back(calculate(*fstIt, *sndIt, *trdIt));
    return DenseElementsAttr::get(fst.getType(), elementResults);
  }
  return {};
}

namespace {

struct constant_int_all_ones_matcher {
  bool match(Operation *op) {
    APInt value;
    return mlir::detail::constant_int_value_binder(&value).match(op) &&
           value.isAllOnes();
  }
};

} // anonymous namespace

unsigned circt::llhd::getLLHDTypeWidth(Type type) {
  if (auto sig = dyn_cast<hw::InOutType>(type))
    type = sig.getElementType();
  else if (auto ptr = dyn_cast<llhd::PtrType>(type))
    type = ptr.getElementType();
  if (auto array = dyn_cast<hw::ArrayType>(type))
    return array.getNumElements();
  if (auto tup = dyn_cast<hw::StructType>(type))
    return tup.getElements().size();
  return type.getIntOrFloatBitWidth();
}

Type circt::llhd::getLLHDElementType(Type type) {
  if (auto sig = dyn_cast<hw::InOutType>(type))
    type = sig.getElementType();
  else if (auto ptr = dyn_cast<llhd::PtrType>(type))
    type = ptr.getElementType();
  if (auto array = dyn_cast<hw::ArrayType>(type))
    return array.getElementType();
  return type;
}

//===---------------------------------------------------------------------===//
// LLHD Operations
//===---------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantTimeOp
//===----------------------------------------------------------------------===//

OpFoldResult llhd::ConstantTimeOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "const has no operands");
  return getValueAttr();
}

void llhd::ConstantTimeOp::build(OpBuilder &builder, OperationState &result,
                                 unsigned time, const StringRef &timeUnit,
                                 unsigned delta, unsigned epsilon) {
  auto *ctx = builder.getContext();
  auto attr = TimeAttr::get(ctx, time, timeUnit, delta, epsilon);
  return build(builder, result, TimeType::get(ctx), attr);
}

//===----------------------------------------------------------------------===//
// SigExtractOp and PtrExtractOp
//===----------------------------------------------------------------------===//

template <class Op>
static OpFoldResult foldSigPtrExtractOp(Op op, ArrayRef<Attribute> operands) {

  if (!operands[1])
    return nullptr;

  // llhd.sig.extract(input, 0) with inputWidth == resultWidth => input
  if (op.getResultWidth() == op.getInputWidth() &&
      cast<IntegerAttr>(operands[1]).getValue().isZero())
    return op.getInput();

  return nullptr;
}

OpFoldResult llhd::SigExtractOp::fold(FoldAdaptor adaptor) {
  return foldSigPtrExtractOp(*this, adaptor.getOperands());
}

OpFoldResult llhd::PtrExtractOp::fold(FoldAdaptor adaptor) {
  return foldSigPtrExtractOp(*this, adaptor.getOperands());
}

//===----------------------------------------------------------------------===//
// SigArraySliceOp and PtrArraySliceOp
//===----------------------------------------------------------------------===//

template <class Op>
static OpFoldResult foldSigPtrArraySliceOp(Op op,
                                           ArrayRef<Attribute> operands) {
  if (!operands[1])
    return nullptr;

  // llhd.sig.array_slice(input, 0) with inputWidth == resultWidth => input
  if (op.getResultWidth() == op.getInputWidth() &&
      cast<IntegerAttr>(operands[1]).getValue().isZero())
    return op.getInput();

  return nullptr;
}

OpFoldResult llhd::SigArraySliceOp::fold(FoldAdaptor adaptor) {
  return foldSigPtrArraySliceOp(*this, adaptor.getOperands());
}

OpFoldResult llhd::PtrArraySliceOp::fold(FoldAdaptor adaptor) {
  return foldSigPtrArraySliceOp(*this, adaptor.getOperands());
}

template <class Op>
static LogicalResult canonicalizeSigPtrArraySliceOp(Op op,
                                                    PatternRewriter &rewriter) {
  IntegerAttr indexAttr;
  if (!matchPattern(op.getLowIndex(), m_Constant(&indexAttr)))
    return failure();

  // llhd.sig.array_slice(llhd.sig.array_slice(target, a), b)
  //   => llhd.sig.array_slice(target, a+b)
  IntegerAttr a;
  if (matchPattern(op.getInput(),
                   m_Op<Op>(matchers::m_Any(), m_Constant(&a)))) {
    auto sliceOp = op.getInput().template getDefiningOp<Op>();
    op.getInputMutable().assign(sliceOp.getInput());
    Value newIndex = rewriter.create<hw::ConstantOp>(
        op->getLoc(), a.getValue() + indexAttr.getValue());
    op.getLowIndexMutable().assign(newIndex);

    return success();
  }

  return failure();
}

LogicalResult llhd::SigArraySliceOp::canonicalize(llhd::SigArraySliceOp op,
                                                  PatternRewriter &rewriter) {
  return canonicalizeSigPtrArraySliceOp(op, rewriter);
}

LogicalResult llhd::PtrArraySliceOp::canonicalize(llhd::PtrArraySliceOp op,
                                                  PatternRewriter &rewriter) {
  return canonicalizeSigPtrArraySliceOp(op, rewriter);
}

//===----------------------------------------------------------------------===//
// SigStructExtractOp and PtrStructExtractOp
//===----------------------------------------------------------------------===//

template <class SigPtrType>
static LogicalResult inferReturnTypesOfStructExtractOp(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  Type type =
      cast<hw::StructType>(
          cast<SigPtrType>(operands[0].getType()).getElementType())
          .getFieldType(
              cast<StringAttr>(attrs.getNamed("field")->getValue()).getValue());
  if (!type) {
    context->getDiagEngine().emit(loc.value_or(UnknownLoc()),
                                  DiagnosticSeverity::Error)
        << "invalid field name specified";
    return failure();
  }
  results.push_back(SigPtrType::get(type));
  return success();
}

LogicalResult llhd::SigStructExtractOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  return inferReturnTypesOfStructExtractOp<hw::InOutType>(
      context, loc, operands, attrs, properties, regions, results);
}

LogicalResult llhd::PtrStructExtractOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  return inferReturnTypesOfStructExtractOp<llhd::PtrType>(
      context, loc, operands, attrs, properties, regions, results);
}

//===----------------------------------------------------------------------===//
// DrvOp
//===----------------------------------------------------------------------===//

LogicalResult llhd::DrvOp::fold(FoldAdaptor adaptor,
                                SmallVectorImpl<OpFoldResult> &result) {
  if (!getEnable())
    return failure();

  if (matchPattern(getEnable(), m_One())) {
    getEnableMutable().clear();
    return success();
  }

  return failure();
}

LogicalResult llhd::DrvOp::canonicalize(llhd::DrvOp op,
                                        PatternRewriter &rewriter) {
  if (!op.getEnable())
    return failure();

  if (matchPattern(op.getEnable(), m_Zero())) {
    rewriter.eraseOp(op);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

// Implement this operation for the BranchOpInterface
SuccessorOperands llhd::WaitOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return SuccessorOperands(getDestOpsMutable());
}

//===----------------------------------------------------------------------===//
// ProcessOp
//===----------------------------------------------------------------------===//

template <typename ModuleTy>
static SmallVector<hw::PortInfo> getPortList(ModuleTy &mod) {
  auto modTy = mod.getHWModuleType();
  auto emptyDict = DictionaryAttr::get(mod.getContext());
  SmallVector<hw::PortInfo> retval;
  auto locs = mod.getAllPortLocs();
  for (unsigned i = 0, e = modTy.getNumPorts(); i < e; ++i) {
    LocationAttr loc = locs[i];
    DictionaryAttr attrs =
        dyn_cast_or_null<DictionaryAttr>(mod.getPortAttrs(i));
    if (!attrs)
      attrs = emptyDict;
    retval.push_back({modTy.getPorts()[i],
                      modTy.isOutput(i) ? modTy.getOutputIdForPortId(i)
                                        : modTy.getInputIdForPortId(i),
                      attrs, loc});
  }
  return retval;
}

template <typename ModuleTy>
static hw::PortInfo getPort(ModuleTy &mod, size_t idx) {
  auto modTy = mod.getHWModuleType();
  auto emptyDict = DictionaryAttr::get(mod.getContext());
  LocationAttr loc = mod.getPortLoc(idx);
  DictionaryAttr attrs =
      dyn_cast_or_null<DictionaryAttr>(mod.getPortAttrs(idx));
  if (!attrs)
    attrs = emptyDict;
  return {modTy.getPorts()[idx],
          modTy.isOutput(idx) ? modTy.getOutputIdForPortId(idx)
                              : modTy.getInputIdForPortId(idx),
          attrs, loc};
}

static bool hasAttribute(StringRef name, ArrayRef<NamedAttribute> attrs) {
  for (auto &argAttr : attrs)
    if (argAttr.getName() == name)
      return true;
  return false;
}

template <typename ModuleTy>
static ParseResult parseHWModuleOp(OpAsmParser &parser,
                                   OperationState &result) {

  using namespace mlir::function_interface_impl;
  auto builder = parser.getBuilder();
  auto loc = parser.getCurrentLocation();

  // Parse the visibility attribute.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the parameters.
  ArrayAttr parameters;
  if (parseOptionalParameterList(parser, parameters))
    return failure();

  SmallVector<hw::module_like_impl::PortParse> ports;
  TypeAttr modType;
  if (failed(
          hw::module_like_impl::parseModuleSignature(parser, ports, modType)))
    return failure();

  // Parse the attribute dict.
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  if (hasAttribute("parameters", result.attributes)) {
    parser.emitError(loc, "explicit `parameters` attributes not allowed");
    return failure();
  }

  result.addAttribute("parameters", parameters);
  result.addAttribute(ModuleTy::getModuleTypeAttrName(result.name), modType);

  // Convert the specified array of dictionary attrs (which may have null
  // entries) to an ArrayAttr of dictionaries.
  SmallVector<Attribute> attrs;
  for (auto &port : ports)
    attrs.push_back(port.attrs ? port.attrs : builder.getDictionaryAttr({}));
  // Add the attributes to the ports.
  auto nonEmptyAttrsFn = [](Attribute attr) {
    return attr && !cast<DictionaryAttr>(attr).empty();
  };
  if (llvm::any_of(attrs, nonEmptyAttrsFn))
    result.addAttribute(ModuleTy::getPerPortAttrsAttrName(result.name),
                        builder.getArrayAttr(attrs));

  // Add the port locations.
  auto unknownLoc = builder.getUnknownLoc();
  auto nonEmptyLocsFn = [unknownLoc](Attribute attr) {
    return attr && cast<Location>(attr) != unknownLoc;
  };
  SmallVector<Attribute> locs;
  StringAttr portLocsAttrName;

  // Plain modules only store the output port locations, as the input port
  // locations will be stored in the basic block arguments.
  portLocsAttrName = ModuleTy::getResultLocsAttrName(result.name);
  for (auto &port : ports)
    if (port.direction == hw::ModulePort::Direction::Output)
      locs.push_back(port.sourceLoc ? Location(*port.sourceLoc) : unknownLoc);

  if (llvm::any_of(locs, nonEmptyLocsFn))
    result.addAttribute(portLocsAttrName, builder.getArrayAttr(locs));

  // Add the entry block arguments.
  SmallVector<OpAsmParser::Argument, 4> entryArgs;
  for (auto &port : ports)
    if (port.direction != hw::ModulePort::Direction::Output)
      entryArgs.push_back(port);

  // Parse the optional function body.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, entryArgs))
    return failure();

  return success();
}

ParseResult ProcessOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseHWModuleOp<ProcessOp>(parser, result);
}

template <typename ModuleTy>
static void printModuleOp(OpAsmPrinter &p, ModuleTy mod) {
  p << ' ';
  // Print the visibility of the module.
  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility = mod.getOperation()->template getAttrOfType<StringAttr>(
          visibilityAttrName))
    p << visibility.getValue() << ' ';

  // Print the operation and the function name.
  p.printSymbolName(SymbolTable::getSymbolName(mod.getOperation()).getValue());

  // Print the parameter list if present.
  printOptionalParameterList(p, mod.getOperation(), mod.getParameters());

  hw::module_like_impl::printModuleSignatureNew(p, mod);

  SmallVector<StringRef, 3> omittedAttrs;
  omittedAttrs.push_back(mod.getResultLocsAttrName());
  omittedAttrs.push_back(mod.getModuleTypeAttrName());
  omittedAttrs.push_back(mod.getPerPortAttrsAttrName());
  omittedAttrs.push_back(mod.getParametersAttrName());
  omittedAttrs.push_back(visibilityAttrName);
  if (auto cmt =
          mod.getOperation()->template getAttrOfType<StringAttr>("comment"))
    if (cmt.getValue().empty())
      omittedAttrs.push_back("comment");

  mlir::function_interface_impl::printFunctionAttributes(p, mod.getOperation(),
                                                         omittedAttrs);
}

void ProcessOp::print(OpAsmPrinter &p) {
  printModuleOp(p, *this);

  // Print the body if this is not an external function.
  Region &body = getBody();
  if (!body.empty()) {
    p << " ";
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

static LogicalResult verifyModuleCommon(hw::HWModuleLike module) {
  assert(isa<hw::HWModuleLike>(module) &&
         "verifier hook should only be called on modules");

  SmallPtrSet<Attribute, 4> paramNames;

  // Check parameter default values are sensible.
  for (auto param : module->getAttrOfType<ArrayAttr>("parameters")) {
    auto paramAttr = cast<hw::ParamDeclAttr>(param);

    // Check that we don't have any redundant parameter names.  These are
    // resolved by string name: reuse of the same name would cause ambiguities.
    if (!paramNames.insert(paramAttr.getName()).second)
      return module->emitOpError("parameter ")
             << paramAttr << " has the same name as a previous parameter";

    // Default values are allowed to be missing, check them if present.
    auto value = paramAttr.getValue();
    if (!value)
      continue;

    auto typedValue = dyn_cast<TypedAttr>(value);
    if (!typedValue)
      return module->emitOpError("parameter ")
             << paramAttr << " should have a typed value; has value " << value;

    if (typedValue.getType() != paramAttr.getType())
      return module->emitOpError("parameter ")
             << paramAttr << " should have type " << paramAttr.getType()
             << "; has type " << typedValue.getType();

    // Verify that this is a valid parameter value, disallowing parameter
    // references.  We could allow parameters to refer to each other in the
    // future with lexical ordering if there is a need.
    if (failed(checkParameterInContext(value, module, module,
                                       /*disallowParamRefs=*/true)))
      return failure();
  }
  return success();
}

LogicalResult ProcessOp::verify() {
  if (failed(verifyModuleCommon(*this)))
    return failure();

  auto type = getModuleType();
  auto *body = getBodyBlock();

  // Verify the number of block arguments.
  auto numInputs = type.getNumInputs();
  if (body->getNumArguments() != numInputs)
    return emitOpError("entry block must have")
           << numInputs << " arguments to match module signature";

  return success();
}

SmallVector<Location> ProcessOp::getAllPortLocs() {
  SmallVector<Location> portLocs;
  portLocs.reserve(getNumPorts());
  auto resultLocs = getResultLocsAttr();
  unsigned inputCount = 0;
  auto modType = getModuleType();
  auto unknownLoc = UnknownLoc::get(getContext());
  auto *body = getBodyBlock();
  for (unsigned i = 0, e = getNumPorts(); i < e; ++i) {
    if (modType.isOutput(i)) {
      auto loc = resultLocs
                     ? cast<Location>(
                           resultLocs.getValue()[portLocs.size() - inputCount])
                     : unknownLoc;
      portLocs.push_back(loc);
    } else {
      auto loc = body ? body->getArgument(inputCount).getLoc() : unknownLoc;
      portLocs.push_back(loc);
      ++inputCount;
    }
  }
  return portLocs;
}

void ProcessOp::setAllPortLocsAttrs(ArrayRef<Attribute> locs) {
  SmallVector<Attribute> resultLocs;
  unsigned inputCount = 0;
  auto modType = getModuleType();
  auto *body = getBodyBlock();
  for (unsigned i = 0, e = getNumPorts(); i < e; ++i) {
    if (modType.isOutput(i))
      resultLocs.push_back(locs[i]);
    else
      body->getArgument(inputCount++).setLoc(cast<Location>(locs[i]));
  }
  setResultLocsAttr(ArrayAttr::get(getContext(), resultLocs));
}

template <typename ModTy>
static void setAllPortNames(ArrayRef<Attribute> names, ModTy module) {
  auto numInputs = module.getNumInputPorts();
  SmallVector<Attribute> argNames(names.begin(), names.begin() + numInputs);
  SmallVector<Attribute> resNames(names.begin() + numInputs, names.end());
  auto oldType = module.getModuleType();
  SmallVector<hw::ModulePort> newPorts(oldType.getPorts().begin(),
                                       oldType.getPorts().end());
  for (size_t i = 0UL, e = newPorts.size(); i != e; ++i)
    newPorts[i].name = cast<StringAttr>(names[i]);
  auto newType = hw::ModuleType::get(module.getContext(), newPorts);
  module.setModuleType(newType);
}

void ProcessOp::setAllPortNames(ArrayRef<Attribute> names) {
  ::setAllPortNames(names, *this);
}

ArrayRef<Attribute> ProcessOp::getAllPortAttrs() {
  auto attrs = getPerPortAttrs();
  if (attrs && !attrs->empty())
    return attrs->getValue();
  return {};
}

static ArrayAttr arrayOrEmpty(mlir::MLIRContext *context,
                              ArrayRef<Attribute> attrs) {
  if (attrs.empty())
    return ArrayAttr::get(context, {});
  bool empty = true;
  for (auto a : attrs)
    if (a && !cast<DictionaryAttr>(a).empty()) {
      empty = false;
      break;
    }
  if (empty)
    return ArrayAttr::get(context, {});
  return ArrayAttr::get(context, attrs);
}

void ProcessOp::setAllPortAttrs(ArrayRef<Attribute> attrs) {
  setPerPortAttrsAttr(arrayOrEmpty(getContext(), attrs));
}

void ProcessOp::removeAllPortAttrs() {
  setPerPortAttrsAttr(ArrayAttr::get(getContext(), {}));
}

template <typename ModTy>
static void setHWModuleType(ModTy &mod, hw::ModuleType type) {
  auto argAttrs = mod.getAllInputAttrs();
  auto resAttrs = mod.getAllOutputAttrs();
  mod.setModuleTypeAttr(TypeAttr::get(type));
  unsigned newNumArgs = type.getNumInputs();
  unsigned newNumResults = type.getNumOutputs();

  auto emptyDict = DictionaryAttr::get(mod.getContext());
  argAttrs.resize(newNumArgs, emptyDict);
  resAttrs.resize(newNumResults, emptyDict);

  SmallVector<Attribute> attrs;
  attrs.append(argAttrs.begin(), argAttrs.end());
  attrs.append(resAttrs.begin(), resAttrs.end());

  if (attrs.empty())
    return mod.removeAllPortAttrs();
  mod.setAllPortAttrs(attrs);
}

void ProcessOp::setHWModuleType(hw::ModuleType type) {
  return ::setHWModuleType(*this, type);
}

template <typename ModuleTy>
static void
buildModule(OpBuilder &builder, OperationState &result, StringAttr name,
            const hw::ModulePortInfo &ports, ArrayAttr parameters,
            ArrayRef<NamedAttribute> attributes, StringAttr comment) {
  using namespace mlir::function_interface_impl;

  // Add an attribute for the name.
  result.addAttribute(SymbolTable::getSymbolAttrName(), name);

  SmallVector<Attribute> perPortAttrs;
  SmallVector<hw::ModulePort> portTypes;

  for (auto elt : ports) {
    portTypes.push_back(elt);
    llvm::SmallVector<NamedAttribute> portAttrs;
    if (elt.attrs)
      llvm::copy(elt.attrs, std::back_inserter(portAttrs));
    perPortAttrs.push_back(builder.getDictionaryAttr(portAttrs));
  }

  // Allow clients to pass in null for the parameters list.
  if (!parameters)
    parameters = builder.getArrayAttr({});

  // Record the argument and result types as an attribute.
  auto type = hw::ModuleType::get(builder.getContext(), portTypes);
  result.addAttribute(ModuleTy::getModuleTypeAttrName(result.name),
                      TypeAttr::get(type));
  result.addAttribute("per_port_attrs",
                      arrayOrEmpty(builder.getContext(), perPortAttrs));
  result.addAttribute("parameters", parameters);
  if (!comment)
    comment = builder.getStringAttr("");
  result.addAttribute("comment", comment);
  result.addAttributes(attributes);
  result.addRegion();
}

void ProcessOp::build(OpBuilder &builder, OperationState &result,
                      StringAttr name, const hw::ModulePortInfo &ports,
                      ArrayAttr parameters, ArrayRef<NamedAttribute> attributes,
                      StringAttr comment, bool shouldEnsureTerminator) {
  buildModule<ProcessOp>(builder, result, name, ports, parameters, attributes,
                         comment);

  // Create a region and a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

  // Add arguments to the body block.
  auto unknownLoc = builder.getUnknownLoc();
  for (auto port : ports.getInputs()) {
    auto loc = port.loc ? Location(port.loc) : unknownLoc;
    auto type = port.type;
    if (port.isInOut() && !isa<hw::InOutType>(type))
      type = hw::InOutType::get(type);
    body->addArgument(type, loc);
  }

  // Add result ports attribute.
  auto unknownLocAttr = cast<LocationAttr>(unknownLoc);
  SmallVector<Attribute> resultLocs;
  for (auto port : ports.getOutputs())
    resultLocs.push_back(port.loc ? port.loc : unknownLocAttr);
  result.addAttribute("result_locs", builder.getArrayAttr(resultLocs));
}

void ProcessOp::build(OpBuilder &builder, OperationState &result,
                      StringAttr name, ArrayRef<hw::PortInfo> ports,
                      ArrayAttr parameters, ArrayRef<NamedAttribute> attributes,
                      StringAttr comment) {
  build(builder, result, name, hw::ModulePortInfo(ports), parameters,
        attributes, comment);
}

//===----------------------------------------------------------------------===//
// ConnectOp
//===----------------------------------------------------------------------===//

LogicalResult llhd::ConnectOp::canonicalize(llhd::ConnectOp op,
                                            PatternRewriter &rewriter) {
  if (op.getLhs() == op.getRhs())
    rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// RegOp
//===----------------------------------------------------------------------===//

ParseResult llhd::RegOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand signal;
  Type signalType;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> valueOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> triggerOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> delayOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> gateOperands;
  SmallVector<Type, 8> valueTypes;
  llvm::SmallVector<int64_t, 8> modesArray;
  llvm::SmallVector<int64_t, 8> gateMask;
  int64_t gateCount = 0;

  if (parser.parseOperand(signal))
    return failure();
  while (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::UnresolvedOperand value;
    OpAsmParser::UnresolvedOperand trigger;
    OpAsmParser::UnresolvedOperand delay;
    OpAsmParser::UnresolvedOperand gate;
    Type valueType;
    StringAttr modeAttr;
    NamedAttrList attrStorage;

    if (parser.parseLParen())
      return failure();
    if (parser.parseOperand(value) || parser.parseComma())
      return failure();
    if (parser.parseAttribute(modeAttr, parser.getBuilder().getNoneType(),
                              "modes", attrStorage))
      return failure();
    auto attrOptional = llhd::symbolizeRegMode(modeAttr.getValue());
    if (!attrOptional)
      return parser.emitError(parser.getCurrentLocation(),
                              "invalid string attribute");
    modesArray.push_back(static_cast<int64_t>(*attrOptional));
    if (parser.parseOperand(trigger))
      return failure();
    if (parser.parseKeyword("after") || parser.parseOperand(delay))
      return failure();
    if (succeeded(parser.parseOptionalKeyword("if"))) {
      gateMask.push_back(++gateCount);
      if (parser.parseOperand(gate))
        return failure();
      gateOperands.push_back(gate);
    } else {
      gateMask.push_back(0);
    }
    if (parser.parseColon() || parser.parseType(valueType) ||
        parser.parseRParen())
      return failure();
    valueOperands.push_back(value);
    triggerOperands.push_back(trigger);
    delayOperands.push_back(delay);
    valueTypes.push_back(valueType);
  }
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(signalType))
    return failure();
  if (parser.resolveOperand(signal, signalType, result.operands))
    return failure();
  if (parser.resolveOperands(valueOperands, valueTypes,
                             parser.getCurrentLocation(), result.operands))
    return failure();
  for (auto operand : triggerOperands)
    if (parser.resolveOperand(operand, parser.getBuilder().getI1Type(),
                              result.operands))
      return failure();
  for (auto operand : delayOperands)
    if (parser.resolveOperand(
            operand, llhd::TimeType::get(parser.getBuilder().getContext()),
            result.operands))
      return failure();
  for (auto operand : gateOperands)
    if (parser.resolveOperand(operand, parser.getBuilder().getI1Type(),
                              result.operands))
      return failure();
  result.addAttribute("gateMask",
                      parser.getBuilder().getI64ArrayAttr(gateMask));
  result.addAttribute("modes", parser.getBuilder().getI64ArrayAttr(modesArray));
  llvm::SmallVector<int32_t, 5> operandSizes;
  operandSizes.push_back(1);
  operandSizes.push_back(valueOperands.size());
  operandSizes.push_back(triggerOperands.size());
  operandSizes.push_back(delayOperands.size());
  operandSizes.push_back(gateOperands.size());
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(operandSizes));

  return success();
}

void llhd::RegOp::print(OpAsmPrinter &printer) {
  printer << " " << getSignal();
  for (size_t i = 0, e = getValues().size(); i < e; ++i) {
    std::optional<llhd::RegMode> mode = llhd::symbolizeRegMode(
        cast<IntegerAttr>(getModes().getValue()[i]).getInt());
    if (!mode) {
      emitError("invalid RegMode");
      return;
    }
    printer << ", (" << getValues()[i] << ", \""
            << llhd::stringifyRegMode(*mode) << "\" " << getTriggers()[i]
            << " after " << getDelays()[i];
    if (hasGate(i))
      printer << " if " << getGateAt(i);
    printer << " : " << getValues()[i].getType() << ")";
  }
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {"modes", "gateMask", "operandSegmentSizes"});
  printer << " : " << getSignal().getType();
}

LogicalResult llhd::RegOp::verify() {
  // At least one trigger has to be present
  if (getTriggers().size() < 1)
    return emitError("At least one trigger quadruple has to be present.");

  // Values variadic operand must have the same size as the triggers variadic
  if (getValues().size() != getTriggers().size())
    return emitOpError("Number of 'values' is not equal to the number of "
                       "'triggers', got ")
           << getValues().size() << " modes, but " << getTriggers().size()
           << " triggers!";

  // Delay variadic operand must have the same size as the triggers variadic
  if (getDelays().size() != getTriggers().size())
    return emitOpError("Number of 'delays' is not equal to the number of "
                       "'triggers', got ")
           << getDelays().size() << " modes, but " << getTriggers().size()
           << " triggers!";

  // Array Attribute of RegModes must have the same number of elements as the
  // variadics
  if (getModes().size() != getTriggers().size())
    return emitOpError("Number of 'modes' is not equal to the number of "
                       "'triggers', got ")
           << getModes().size() << " modes, but " << getTriggers().size()
           << " triggers!";

  // Array Attribute 'gateMask' must have the same number of elements as the
  // triggers and values variadics
  if (getGateMask().size() != getTriggers().size())
    return emitOpError("Size of 'gateMask' is not equal to the size of "
                       "'triggers', got ")
           << getGateMask().size() << " modes, but " << getTriggers().size()
           << " triggers!";

  // Number of non-zero elements in 'gateMask' has to be the same as the size
  // of the gates variadic, also each number from 1 to size-1 has to occur
  // only once and in increasing order
  unsigned counter = 0;
  unsigned prevElement = 0;
  for (Attribute maskElem : getGateMask().getValue()) {
    int64_t val = cast<IntegerAttr>(maskElem).getInt();
    if (val < 0)
      return emitError("Element in 'gateMask' must not be negative!");
    if (val == 0)
      continue;
    if (val != ++prevElement)
      return emitError(
          "'gateMask' has to contain every number from 1 to the "
          "number of gates minus one exactly once in increasing order "
          "(may have zeros in-between).");
    counter++;
  }
  if (getGates().size() != counter)
    return emitError("The number of non-zero elements in 'gateMask' and the "
                     "size of the 'gates' variadic have to match.");

  // Each value must be either the same type as the 'signal' or the underlying
  // type of the 'signal'
  for (auto val : getValues()) {
    if (val.getType() != getSignal().getType() &&
        val.getType() !=
            cast<hw::InOutType>(getSignal().getType()).getElementType()) {
      return emitOpError(
          "type of each 'value' has to be either the same as the "
          "type of 'signal' or the underlying type of 'signal'");
    }
  }
  return success();
}

#include "circt/Dialect/LLHD/IR/LLHDEnums.cpp.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/LLHD/IR/LLHD.cpp.inc"
