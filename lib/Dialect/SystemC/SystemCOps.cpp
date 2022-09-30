//===- SystemCOps.cpp - Implement the SystemC operations ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SystemC ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SystemC/SystemCOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/ModuleImplementation.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace circt::systemc;

//===----------------------------------------------------------------------===//
// ImplicitSSAName Custom Directive
//===----------------------------------------------------------------------===//

static ParseResult parseImplicitSSAName(OpAsmParser &parser,
                                        StringAttr &nameAttr) {
  nameAttr = parser.getBuilder().getStringAttr(parser.getResultName(0).first);
  return success();
}

static void printImplicitSSAName(OpAsmPrinter &p, Operation *op,
                                 StringAttr nameAttr) {}

//===----------------------------------------------------------------------===//
// SCModuleOp
//===----------------------------------------------------------------------===//

static hw::PortDirection getDirection(Type type) {
  return TypeSwitch<Type, hw::PortDirection>(type)
      .Case<InOutType>([](auto ty) { return hw::PortDirection::INOUT; })
      .Case<InputType>([](auto ty) { return hw::PortDirection::INPUT; })
      .Case<OutputType>([](auto ty) { return hw::PortDirection::OUTPUT; });
}

SCModuleOp::PortDirectionRange
SCModuleOp::getPortsOfDirection(hw::PortDirection direction) {
  std::function<bool(const BlockArgument &)> predicateFn =
      [&](const BlockArgument &arg) -> bool {
    return getDirection(arg.getType()) == direction;
  };
  return llvm::make_filter_range(getArguments(), predicateFn);
}

void SCModuleOp::getPortInfoList(SmallVectorImpl<hw::PortInfo> &portInfoList) {
  for (int i = 0, e = getNumArguments(); i < e; ++i) {
    hw::PortInfo info;
    info.name = getPortNames()[i].cast<StringAttr>();
    info.type = getSignalBaseType(getArgument(i).getType());
    info.direction = getDirection(info.type);
    portInfoList.push_back(info);
  }
}

mlir::Region *SCModuleOp::getCallableRegion() { return &getBody(); }

ArrayRef<mlir::Type> SCModuleOp::getCallableResults() {
  return getResultTypes();
}

StringRef SCModuleOp::getModuleName() {
  return (*this)
      ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
      .getValue();
}

ParseResult SCModuleOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr moduleName;
  SmallVector<OpAsmParser::Argument, 4> args;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;
  SmallVector<Attribute> argNames;
  SmallVector<DictionaryAttr> resultAttrs;

  // Parse the visibility attribute.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  if (parser.parseSymbolName(moduleName, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  bool isVariadic = false;
  if (hw::module_like_impl::parseModuleFunctionSignature(
          parser, args, isVariadic, resultTypes, resultAttrs, argNames))
    return failure();

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  for (auto &arg : args) {
    argNames.push_back(
        StringAttr::get(parser.getContext(), arg.ssaName.name.drop_front()));
    argTypes.push_back(arg.type);
  }

  result.addAttribute("portNames",
                      ArrayAttr::get(parser.getContext(), argNames));

  auto type = parser.getBuilder().getFunctionType(argTypes, resultTypes);
  result.addAttribute(SCModuleOp::getTypeAttrName(), TypeAttr::get(type));

  mlir::function_interface_impl::addArgAndResultAttrs(
      parser.getBuilder(), result, args, resultAttrs);

  auto &body = *result.addRegion();
  if (parser.parseRegion(body, args))
    return failure();
  if (body.empty())
    body.push_back(std::make_unique<Block>().release());

  return success();
}

void SCModuleOp::print(OpAsmPrinter &p) {
  p << ' ';

  // Print the visibility of the module.
  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility =
          getOperation()->getAttrOfType<StringAttr>(visibilityAttrName))
    p << visibility.getValue() << ' ';

  p.printSymbolName(SymbolTable::getSymbolName(*this).getValue());
  p << ' ';

  bool needArgNamesAttr = false;
  hw::module_like_impl::printModuleSignature(
      p, *this, getFunctionType().getInputs(), false, {}, needArgNamesAttr);
  mlir::function_interface_impl::printFunctionAttributes(
      p, *this, getFunctionType().getInputs().size(), 0, {"portNames"});

  p << ' ';
  p.printRegion(getBody(), false, false);
}

static Type wrapPortType(Type type, hw::PortDirection direction) {
  if (auto inoutTy = type.dyn_cast<hw::InOutType>())
    type = inoutTy.getElementType();

  switch (direction) {
  case hw::PortDirection::INOUT:
    return InOutType::get(type);
  case hw::PortDirection::INPUT:
    return InputType::get(type);
  case hw::PortDirection::OUTPUT:
    return OutputType::get(type);
  }
  llvm_unreachable("Impossible port direction");
}

void SCModuleOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       StringAttr name, ArrayAttr portNames,
                       ArrayRef<Type> portTypes,
                       ArrayRef<NamedAttribute> attributes) {
  odsState.addAttribute(getPortNamesAttrName(odsState.name), portNames);
  Region *region = odsState.addRegion();

  auto moduleType = odsBuilder.getFunctionType(portTypes, {});
  odsState.addAttribute(getTypeAttrName(), TypeAttr::get(moduleType));

  odsState.addAttribute(SymbolTable::getSymbolAttrName(), name);
  region->push_back(new Block);
  region->addArguments(
      portTypes,
      SmallVector<Location>(portTypes.size(), odsBuilder.getUnknownLoc()));
  odsState.addAttributes(attributes);
}

void SCModuleOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       StringAttr name, ArrayRef<hw::PortInfo> ports,
                       ArrayRef<NamedAttribute> attributes) {
  MLIRContext *ctxt = odsBuilder.getContext();
  SmallVector<Attribute> portNames;
  SmallVector<Type> portTypes;
  for (auto port : ports) {
    portNames.push_back(StringAttr::get(ctxt, port.getName()));
    portTypes.push_back(wrapPortType(port.type, port.direction));
  }
  build(odsBuilder, odsState, name, ArrayAttr::get(ctxt, portNames), portTypes);
}

void SCModuleOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       StringAttr name, const hw::ModulePortInfo &ports,
                       ArrayRef<NamedAttribute> attributes) {
  SmallVector<hw::PortInfo> portInfos(ports.inputs);
  portInfos.append(ports.outputs);
  build(odsBuilder, odsState, name, portInfos, attributes);
}

void SCModuleOp::getAsmBlockArgumentNames(mlir::Region &region,
                                          mlir::OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;

  ArrayAttr portNames = getPortNames();
  for (size_t i = 0, e = getNumArguments(); i != e; ++i) {
    auto str = portNames[i].cast<StringAttr>().getValue();
    setNameFn(getArgument(i), str);
  }
}

LogicalResult SCModuleOp::verify() {
  if (getFunctionType().getNumResults() != 0)
    return emitOpError(
        "incorrect number of function results (always has to be 0)");
  if (getPortNames().size() != getFunctionType().getNumInputs())
    return emitOpError("incorrect number of port names");

  for (auto arg : getArguments()) {
    if (!hw::type_isa<InputType, OutputType, InOutType>(arg.getType()))
      return mlir::emitError(
          arg.getLoc(),
          "module port must be of type 'sc_in', 'sc_out', or 'sc_inout'");
  }

  for (auto portName : getPortNames()) {
    if (portName.cast<StringAttr>().getValue().empty())
      return emitOpError("port name must not be empty");
  }

  return success();
}

LogicalResult SCModuleOp::verifyRegions() {
  DenseMap<StringRef, BlockArgument> portNames;
  DenseMap<StringRef, Operation *> memberNames;
  DenseMap<StringRef, Operation *> localNames;

  bool portsVerified = true;

  for (auto arg : llvm::zip(getPortNames(), getArguments())) {
    StringRef argName = std::get<0>(arg).cast<StringAttr>().getValue();
    BlockArgument argValue = std::get<1>(arg);

    if (portNames.count(argName)) {
      auto diag = mlir::emitError(argValue.getLoc(), "redefines port name '")
                  << argName << "'";
      diag.attachNote(portNames[argName].getLoc())
          << "'" << argName << "' first defined here";
      diag.attachNote(getLoc()) << "in module '@" << getModuleName() << "'";
      portsVerified = false;
      continue;
    }

    portNames.insert({argName, argValue});
  }

  WalkResult result = walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<SCModuleOp>(op->getParentOp()))
      localNames.clear();

    if (auto nameDeclOp = dyn_cast<SystemCNameDeclOpInterface>(op)) {
      StringRef name = nameDeclOp.getName();

      auto reportNameRedefinition = [&](Location firstLoc) -> WalkResult {
        auto diag = mlir::emitError(op->getLoc(), "redefines name '")
                    << name << "'";
        diag.attachNote(firstLoc) << "'" << name << "' first defined here";
        diag.attachNote(getLoc()) << "in module '@" << getModuleName() << "'";
        return WalkResult::interrupt();
      };

      if (portNames.count(name))
        return reportNameRedefinition(portNames[name].getLoc());
      if (memberNames.count(name))
        return reportNameRedefinition(memberNames[name]->getLoc());
      if (localNames.count(name))
        return reportNameRedefinition(localNames[name]->getLoc());

      if (isa<SCModuleOp>(op->getParentOp()))
        memberNames.insert({name, op});
      else
        localNames.insert({name, op});
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted() || !portsVerified)
    return failure();

  return success();
}

CtorOp SCModuleOp::getOrCreateCtor() {
  CtorOp ctor;
  getBody().walk([&](Operation *op) {
    if ((ctor = dyn_cast<CtorOp>(op)))
      return WalkResult::interrupt();

    return WalkResult::skip();
  });

  if (ctor)
    return ctor;

  return OpBuilder(getBody()).create<CtorOp>(getLoc());
}

DestructorOp SCModuleOp::getOrCreateDestructor() {
  DestructorOp destructor;
  getBody().walk([&](Operation *op) {
    if ((destructor = dyn_cast<DestructorOp>(op)))
      return WalkResult::interrupt();

    return WalkResult::skip();
  });

  if (destructor)
    return destructor;

  return OpBuilder::atBlockEnd(getBodyBlock()).create<DestructorOp>(getLoc());
}

//===----------------------------------------------------------------------===//
// SignalOp
//===----------------------------------------------------------------------===//

void SignalOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getSignal(), getName());
}

//===----------------------------------------------------------------------===//
// CtorOp
//===----------------------------------------------------------------------===//

LogicalResult CtorOp::verify() {
  if (getBody().getNumArguments() != 0)
    return emitOpError("must not have any arguments");

  return success();
}

//===----------------------------------------------------------------------===//
// SCFuncOp
//===----------------------------------------------------------------------===//

void SCFuncOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getHandle(), getName());
}

LogicalResult SCFuncOp::verify() {
  if (getBody().getNumArguments() != 0)
    return emitOpError("must not have any arguments");

  return success();
}

//===----------------------------------------------------------------------===//
// InstanceDeclOp
//===----------------------------------------------------------------------===//

void InstanceDeclOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getInstanceHandle(), getName());
}

Operation *InstanceDeclOp::getReferencedModule(const hw::HWSymbolCache *cache) {
  if (cache)
    if (auto *result = cache->getDefinition(getModuleNameAttr()))
      return result;

  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  return topLevelModuleOp.lookupSymbol(getModuleName());
}

Operation *InstanceDeclOp::getReferencedModule() {
  return getReferencedModule(/*cache=*/nullptr);
}

LogicalResult
InstanceDeclOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *module =
      symbolTable.lookupNearestSymbolFrom(*this, getModuleNameAttr());
  if (module == nullptr)
    return emitError("cannot find module definition '")
           << getModuleName() << "'";

  auto emitError = [&](const std::function<void(InFlightDiagnostic & diag)> &fn)
      -> LogicalResult {
    auto diag = emitOpError();
    fn(diag);
    diag.attachNote(module->getLoc()) << "module declared here";
    return failure();
  };

  // It must be a systemc module.
  if (!isa<SCModuleOp>(module))
    return emitError([&](auto &diag) {
      diag << "symbol reference '" << getModuleName()
           << "' isn't a systemc module";
    });

  auto scModule = cast<SCModuleOp>(module);

  // Check that the module name of the symbol and instance type match.
  if (scModule.getModuleName() != getInstanceType().getModuleName())
    return emitError([&](auto &diag) {
      diag << "module names must match; expected '" << scModule.getModuleName()
           << "' but got '" << getInstanceType().getModuleName().getValue()
           << "'";
    });

  // Check that port types and names are consistent with the referenced module.
  ArrayRef<ModuleType::PortInfo> ports = getInstanceType().getPorts();
  ArrayAttr modArgNames = scModule.getPortNames();
  auto numPorts = ports.size();
  auto expectedPortTypes = scModule.getArgumentTypes();

  if (expectedPortTypes.size() != numPorts)
    return emitError([&](auto &diag) {
      diag << "has a wrong number of ports; expected "
           << expectedPortTypes.size() << " but got " << numPorts;
    });

  for (size_t i = 0; i != numPorts; ++i) {
    if (ports[i].type != expectedPortTypes[i]) {
      return emitError([&](auto &diag) {
        diag << "port type #" << i << " must be " << expectedPortTypes[i]
             << ", but got " << ports[i].type;
      });
    }

    if (ports[i].name != modArgNames[i])
      return emitError([&](auto &diag) {
        diag << "port name #" << i << " must be " << modArgNames[i]
             << ", but got " << ports[i].name;
      });
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DestructorOp
//===----------------------------------------------------------------------===//

LogicalResult DestructorOp::verify() {
  if (getBody().getNumArguments() != 0)
    return emitOpError("must not have any arguments");

  return success();
}

//===----------------------------------------------------------------------===//
// ExternOp
//===----------------------------------------------------------------------===//

LogicalResult ExternOp::verify() {
  if (getBody().getNumArguments() != 0)
    return emitOpError("must not have any arguments");

  return success();
}

//===----------------------------------------------------------------------===//
// BindPortOp
//===----------------------------------------------------------------------===//

ParseResult BindPortOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand instance, channel;
  std::string portName;
  if (parser.parseOperand(instance) || parser.parseLSquare() ||
      parser.parseString(&portName))
    return failure();

  auto portNameLoc = parser.getCurrentLocation();

  if (parser.parseRSquare() || parser.parseKeyword("to") ||
      parser.parseOperand(channel))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  auto typeListLoc = parser.getCurrentLocation();
  SmallVector<Type> types;
  if (parser.parseColonTypeList(types))
    return failure();

  if (types.size() != 2)
    return parser.emitError(typeListLoc,
                            "expected a list of exactly 2 types, but got ")
           << types.size();

  if (parser.resolveOperand(instance, types[0], result.operands))
    return failure();
  if (parser.resolveOperand(channel, types[1], result.operands))
    return failure();

  if (auto moduleType = types[0].dyn_cast<ModuleType>()) {
    auto ports = moduleType.getPorts();
    uint64_t index = 0;
    for (auto port : ports) {
      if (port.name == portName)
        break;
      index++;
    }
    if (index >= ports.size())
      return parser.emitError(portNameLoc, "port name \"")
             << portName << "\" not found in module";

    result.addAttribute("portId", parser.getBuilder().getIndexAttr(index));

    return success();
  }

  return failure();
}

void BindPortOp::print(OpAsmPrinter &p) {
  p << " " << getInstance() << "["
    << getInstance()
           .getType()
           .cast<ModuleType>()
           .getPorts()[getPortId().getZExtValue()]
           .name
    << "] to " << getChannel();
  p.printOptionalAttrDict((*this)->getAttrs(), {"portId"});
  p << " : " << getInstance().getType() << ", " << getChannel().getType();
}

LogicalResult BindPortOp::verify() {
  auto ports = getInstance().getType().cast<ModuleType>().getPorts();
  if (getPortId().getZExtValue() >= ports.size())
    return emitOpError("port #")
           << getPortId().getZExtValue() << " does not exist, there are only "
           << ports.size() << " ports";

  // Verify that the base types match.
  Type portType = ports[getPortId().getZExtValue()].type;
  Type channelType = getChannel().getType();
  if (getSignalBaseType(portType) != getSignalBaseType(channelType))
    return emitOpError() << portType << " port cannot be bound to "
                         << channelType << " channel due to base type mismatch";

  // Verify that the port/channel directions are valid.
  if ((portType.isa<InputType>() && channelType.isa<OutputType>()) ||
      (portType.isa<OutputType>() && channelType.isa<InputType>()))
    return emitOpError() << portType << " port cannot be bound to "
                         << channelType
                         << " channel due to port direction mismatch";

  return success();
}

StringRef BindPortOp::getPortName() {
  return getInstance()
      .getType()
      .cast<ModuleType>()
      .getPorts()[getPortId().getZExtValue()]
      .name.getValue();
}

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

void VariableOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getVariable(), getName());
}

ParseResult VariableOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (parseImplicitSSAName(parser, nameAttr))
    return failure();
  result.addAttribute("name", nameAttr);

  OpAsmParser::UnresolvedOperand init;
  auto initResult = parser.parseOptionalOperand(init);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  Type variableType;
  if (parser.parseColonType(variableType))
    return failure();

  if (initResult.has_value()) {
    if (parser.resolveOperand(init, variableType, result.operands))
      return failure();
  }
  result.addTypes({variableType});

  return success();
}

void VariableOp::print(::mlir::OpAsmPrinter &p) {
  p << " ";

  if (getInit())
    p << getInit() << " ";

  p.printOptionalAttrDict(getOperation()->getAttrs(), {"name"});
  p << ": " << getVariable().getType();
}

LogicalResult VariableOp::verify() {
  if (getInit() && getInit().getType() != getVariable().getType())
    return emitOpError(
               "'init' and 'variable' must have the same type, but got ")
           << getInit().getType() << " and " << getVariable().getType();

  return success();
}

//===----------------------------------------------------------------------===//
// InteropVerilatedOp
//===----------------------------------------------------------------------===//

/// Create a instance that refers to a known module.
void InteropVerilatedOp::build(OpBuilder &builder, OperationState &result,
                               Operation *module, StringAttr name,
                               ArrayRef<Value> inputs) {
  assert(hw::isAnyModule(module) && "Can only reference a module");

  FunctionType modType = hw::getModuleType(module);
  build(builder, result, modType.getResults(), name,
        FlatSymbolRefAttr::get(SymbolTable::getSymbolName(module)),
        module->getAttrOfType<ArrayAttr>("argNames"),
        module->getAttrOfType<ArrayAttr>("resultNames"), inputs);
}

/// Lookup the module or extmodule for the symbol.  This returns null on
/// invalid IR.
Operation *
InteropVerilatedOp::getReferencedModule(const hw::HWSymbolCache *cache) {
  if (cache)
    if (auto *result = cache->getDefinition(getModuleNameAttr()))
      return result;

  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  return topLevelModuleOp.lookupSymbol(getModuleName());
}

Operation *InteropVerilatedOp::getReferencedModule() {
  return getReferencedModule(/*cache=*/nullptr);
}

LogicalResult
InteropVerilatedOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *module =
      symbolTable.lookupNearestSymbolFrom(*this, getModuleNameAttr());
  if (module == nullptr)
    return emitError("Cannot find module definition '")
           << getModuleName() << "'";

  // It must be some sort of module.
  if (!hw::isAnyModule(module))
    return emitError("symbol reference '")
           << getModuleName() << "' isn't a module";

  // Check that input and result types are consistent with the referenced
  // module.
  // Emit an error message on the instance, with a note indicating which module
  // is being referenced.
  auto emitError =
      [&](std::function<void(InFlightDiagnostic & diag)> fn) -> LogicalResult {
    auto diag = emitOpError();
    fn(diag);
    diag.attachNote(module->getLoc()) << "module declared here";
    return failure();
  };

  // Make sure our port and result names match.
  ArrayAttr argNames = getInputNamesAttr();
  ArrayAttr modArgNames = module->getAttrOfType<ArrayAttr>("argNames");

  // Check operand types first.
  auto numOperands = getOperation()->getNumOperands();
  auto expectedOperandTypes = hw::getModuleType(module).getInputs();

  if (expectedOperandTypes.size() != numOperands)
    return emitError([&](auto &diag) {
      diag << "has a wrong number of operands; expected "
           << expectedOperandTypes.size() << " but got " << numOperands;
    });

  if (argNames.size() != numOperands)
    return emitError([&](auto &diag) {
      diag << "has a wrong number of input port names; expected " << numOperands
           << " but got " << argNames.size();
    });

  for (size_t i = 0; i != numOperands; ++i) {
    auto expectedType = expectedOperandTypes[i];

    auto operandType = getOperand(i).getType();
    if (operandType != expectedType) {
      return emitError([&](auto &diag) {
        diag << "operand type #" << i << " must be " << expectedType
             << ", but got " << operandType;
      });
    }

    if (argNames[i] != modArgNames[i])
      return emitError([&](auto &diag) {
        diag << "input label #" << i << " must be " << modArgNames[i]
             << ", but got " << argNames[i];
      });
  }

  // Check result types and labels.
  auto numResults = getOperation()->getNumResults();
  auto expectedResultTypes = hw::getModuleType(module).getResults();
  ArrayAttr resultNames = getResultNamesAttr();
  ArrayAttr modResultNames = module->getAttrOfType<ArrayAttr>("resultNames");

  if (expectedResultTypes.size() != numResults)
    return emitError([&](auto &diag) {
      diag << "has a wrong number of results; expected "
           << expectedResultTypes.size() << " but got " << numResults;
    });
  if (resultNames.size() != numResults)
    return emitError([&](auto &diag) {
      diag << "has a wrong number of results port labels; expected "
           << numResults << " but got " << resultNames.size();
    });

  for (size_t i = 0; i != numResults; ++i) {
    auto expectedType = expectedResultTypes[i];

    auto resultType = getResult(i).getType();
    if (resultType != expectedType)
      return emitError([&](auto &diag) {
        diag << "result type #" << i << " must be " << expectedType
             << ", but got " << resultType;
      });

    if (resultNames[i] != modResultNames[i])
      return emitError([&](auto &diag) {
        diag << "input label #" << i << " must be " << modResultNames[i]
             << ", but got " << resultNames[i];
      });
  }

  return success();
}

ParseResult InteropVerilatedOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  auto *context = result.getContext();
  StringAttr instanceNameAttr;
  FlatSymbolRefAttr moduleNameAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands;
  SmallVector<Type> inputsTypes;
  SmallVector<Type> allResultTypes;
  SmallVector<Attribute> inputNames, resultNames;
  auto noneType = parser.getBuilder().getType<NoneType>();

  if (parser.parseAttribute(instanceNameAttr, noneType, "instanceName",
                            result.attributes))
    return failure();

  auto parseInputPort = [&]() -> ParseResult {
    std::string portName;
    if (parser.parseKeywordOrString(&portName))
      return failure();
    inputNames.push_back(StringAttr::get(context, portName));
    inputsOperands.push_back({});
    inputsTypes.push_back({});
    return failure(parser.parseColon() ||
                   parser.parseOperand(inputsOperands.back()) ||
                   parser.parseColon() || parser.parseType(inputsTypes.back()));
  };

  auto parseResultPort = [&]() -> ParseResult {
    std::string portName;
    if (parser.parseKeywordOrString(&portName))
      return failure();
    resultNames.push_back(StringAttr::get(parser.getContext(), portName));
    allResultTypes.push_back({});
    return parser.parseColonType(allResultTypes.back());
  };

  llvm::SMLoc inputsOperandsLoc;
  if (parser.parseAttribute(moduleNameAttr, noneType, "moduleName",
                            result.attributes) ||
      parser.getCurrentLocation(&inputsOperandsLoc) ||
      parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseInputPort) ||
      parser.resolveOperands(inputsOperands, inputsTypes, inputsOperandsLoc,
                             result.operands) ||
      parser.parseArrow() ||
      parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseResultPort) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  result.addAttribute("inputNames",
                      parser.getBuilder().getArrayAttr(inputNames));
  result.addAttribute("resultNames",
                      parser.getBuilder().getArrayAttr(resultNames));
  result.addTypes(allResultTypes);
  return success();
}

void InteropVerilatedOp::print(OpAsmPrinter &p) {
  // hw::ModulePortInfo portInfo = hw::getModulePortInfo(*this);
  size_t nextInputPort = 0, nextOutputPort = 0;

  auto printPortName = [&](size_t &nextPort, ArrayAttr portList) {
    // Allow printing mangled instances.
    if (nextPort >= portList.size()) {
      p << "<corrupt port>: ";
      return;
    }

    p.printKeywordOrString(portList[nextPort++].cast<StringAttr>().getValue());
    p << ": ";
  };

  p << ' ';
  p.printAttributeWithoutType(getInstanceNameAttr());
  p << ' ';
  p.printAttributeWithoutType(getModuleNameAttr());
  p << '(';
  llvm::interleaveComma(getInputs(), p, [&](Value op) {
    // printPortName(nextInputPort, portInfo.inputs);
    printPortName(nextInputPort, getInputNames());
    p << op << ": " << op.getType();
  });
  p << ") -> (";
  llvm::interleaveComma(getResults(), p, [&](Value res) {
    printPortName(nextOutputPort, getResultNames());
    p << res.getType();
  });
  p << ')';
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"instanceName", "moduleName",
                                           "inputNames", "resultNames"});
}

/// Return the name of the specified input port or null if it cannot be
/// determined.
StringAttr InteropVerilatedOp::getArgumentName(size_t idx) {
  auto names = getInputNames();
  // Tolerate malformed IR here to enable debug printing etc.
  if (names && idx < names.size())
    return names[idx].cast<StringAttr>();
  return StringAttr();
}

/// Return the name of the specified result or null if it cannot be
/// determined.
StringAttr InteropVerilatedOp::getResultName(size_t idx) {
  auto names = getResultNames();
  // Tolerate malformed IR here to enable debug printing etc.
  if (names && idx < names.size())
    return names[idx].cast<StringAttr>();
  return StringAttr();
}

/// Change the name of the specified input port.
void InteropVerilatedOp::setArgumentName(size_t i, StringAttr name) {
  auto names = getInputNames();
  SmallVector<Attribute> newNames(names.begin(), names.end());
  if (newNames[i] == name)
    return;
  newNames[i] = name;
  setArgumentNames(ArrayAttr::get(getContext(), names));
}

/// Change the name of the specified output port.
void InteropVerilatedOp::setResultName(size_t i, StringAttr name) {
  auto names = getResultNames();
  SmallVector<Attribute> newNames(names.begin(), names.end());
  if (newNames[i] == name)
    return;
  newNames[i] = name;
  setResultNames(ArrayAttr::get(getContext(), names));
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void InteropVerilatedOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // Provide default names for instance results.
  std::string name = getInstanceName().str() + ".";
  size_t baseNameLen = name.size();

  for (size_t i = 0, e = getNumResults(); i != e; ++i) {
    auto resName = getResultName(i);
    name.resize(baseNameLen);
    if (resName && !resName.getValue().empty())
      name += resName.getValue().str();
    else
      name += std::to_string(i);
    setNameFn(getResult(i), name);
  }
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/SystemC/SystemC.cpp.inc"
