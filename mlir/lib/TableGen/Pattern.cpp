//===- Pattern.cpp - Pattern wrapper class --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pattern wrapper class to simplify using TableGen Record defining a MLIR
// Pattern.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "mlir/TableGen/Pattern.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "mlir/TableGen/AttrOrTypeDef.h"

#define DEBUG_TYPE "mlir-tblgen-pattern"

using namespace mlir;
using namespace tblgen;

using llvm::formatv;

//===----------------------------------------------------------------------===//
// DagLeaf
//===----------------------------------------------------------------------===//

bool DagLeaf::isUnspecified() const {
  return isa_and_nonnull<llvm::UnsetInit>(def);
}

bool DagLeaf::isTypeConstraint() const {
  // Operand matchers specify a type constraint.
  return isSubClassOf("TypeConstraint");
}

bool DagLeaf::isAttrConstraint() const {
  // Attribute matchers specify an attribute constraint.
  return isSubClassOf("AttrConstraint");
}

bool DagLeaf::isNativeCodeCall() const {
  return isSubClassOf("NativeCodeCall");
}

bool DagLeaf::isConstantAttr() const { return isSubClassOf("ConstantAttr"); }

bool DagLeaf::isEnumAttrCase() const {
  return isSubClassOf("EnumAttrCaseInfo");
}

bool DagLeaf::isInt() const { return isa<llvm::IntInit>(def); }


bool DagLeaf::isString() const { return isa<llvm::StringInit>(def); }

int64_t DagLeaf::getIntValue() const {
  assert(isInt() && "the DAG leaf must be an IntegerAttribute");
  return llvm::dyn_cast_or_null<llvm::IntInit>(def)->getValue();
}

Constraint DagLeaf::getAsConstraint() const {
  assert((isTypeConstraint() || isAttrConstraint()) &&
         "the DAG leaf must be operand or attribute");
  return Constraint(cast<llvm::DefInit>(def)->getDef());
}

ConstantAttr DagLeaf::getAsConstantAttr() const {
  assert(isConstantAttr() && "the DAG leaf must be constant attribute");
  return ConstantAttr(cast<llvm::DefInit>(def));
}

EnumAttrCase DagLeaf::getAsEnumAttrCase() const {
  assert(isEnumAttrCase() && "the DAG leaf must be an enum attribute case");
  return EnumAttrCase(cast<llvm::DefInit>(def));
}

std::string DagLeaf::getConditionTemplate() const {
  return getAsConstraint().getConditionTemplate();
}

llvm::StringRef DagLeaf::getNativeCodeTemplate() const {
  assert(isNativeCodeCall() && "the DAG leaf must be NativeCodeCall");
  return cast<llvm::DefInit>(def)->getDef()->getValueAsString("expression");
}

int DagLeaf::getNumReturnsOfNativeCode() const {
  assert(isNativeCodeCall() && "the DAG leaf must be NativeCodeCall");
  return cast<llvm::DefInit>(def)->getDef()->getValueAsInt("numReturns");
}

std::string DagLeaf::getStringAttr() const {
  assert(isString() && "the DAG leaf must be string attribute");
  return def->getAsUnquotedString();
}
bool DagLeaf::isSubClassOf(StringRef superclass) const {
  if (auto *defInit = dyn_cast_or_null<llvm::DefInit>(def))
    return defInit->getDef()->isSubClassOf(superclass);
  return false;
}

void DagLeaf::print(raw_ostream &os) const {
  if (def)
    def->print(os);
}

//===----------------------------------------------------------------------===//
// DagNode
//===----------------------------------------------------------------------===//

bool DagNode::isNativeCodeCall() const {
  if (auto *defInit = dyn_cast_or_null<llvm::DefInit>(node->getOperator()))
    return defInit->getDef()->isSubClassOf("NativeCodeCall");
  return false;
}

bool DagNode::isOperation() const {
  return !isComplexTypeConstraint() && !isNativeCodeCall() &&
         !isReplaceWithValue() && !isLocationDirective() &&
         !isReturnTypeDirective() && !isEither() && !isRepeat() && !isYield();
}

bool DagNode::isComplexTypeConstraint() const {
  if (auto *defInit = dyn_cast_or_null<llvm::DefInit>(node->getOperator()))
    return defInit->getDef()->isSubClassOf("TypeConstraint");
  return false;
}

bool DagNode::isTypeConstraint() const {
  // Operand matchers specify a type constraint.
  if (auto *defInit = dyn_cast_or_null<llvm::DefInit>(node))
    return defInit->getDef()->isSubClassOf("TypeConstraint");
  return false;
}

bool DagNode::isAttrConstraint() const {
  // Attribute matchers specify an attribute constraint.
  if (auto *defInit = dyn_cast_or_null<llvm::DefInit>(node->getOperator()))
    return defInit->getDef()->isSubClassOf("AttrConstraint");
  return false;
}

Constraint DagNode::getAsConstraint() const {
  assert((isTypeConstraint() || isAttrConstraint() || isComplexTypeConstraint()) &&
         "the DAG node must be operand or attribute");
  return Constraint(cast<llvm::DefInit>(node->getOperator())->getDef());
}

llvm::StringRef DagNode::getNativeCodeTemplate() const {
  assert(isNativeCodeCall() && "the DAG leaf must be NativeCodeCall");
  return cast<llvm::DefInit>(node->getOperator())
      ->getDef()
      ->getValueAsString("expression");
}

int DagNode::getNumReturnsOfNativeCode() const {
  assert(isNativeCodeCall() && "the DAG leaf must be NativeCodeCall");
  return cast<llvm::DefInit>(node->getOperator())
      ->getDef()
      ->getValueAsInt("numReturns");
}

llvm::StringRef DagNode::getSymbol() const { return node->getNameStr(); }

Operator &DagNode::getDialectOp(RecordOperatorMap *mapper) const {
  llvm::Record *opDef = cast<llvm::DefInit>(node->getOperator())->getDef();
  auto it = mapper->find(opDef);
  if (it != mapper->end())
    return *it->second;
  return *mapper->try_emplace(opDef, std::make_unique<Operator>(opDef))
              .first->second;
}

int DagNode::getNumOps() const {
  // We want to get number of operations recursively involved in the DAG tree.
  // All other directives should be excluded.
  int count = isOperation() ? 1 : 0;
  for (int i = 0, e = getNumArgs(); i != e; ++i) {
    if (auto child = getArgAsDagNode(i))
      count += child.getNumOps();
  }
  return count;
}

int DagNode::getNumArgs() const { return node->getNumArgs(); }

bool DagNode::isNestedDagArg(unsigned index) const {
  return isa<llvm::DagInit>(node->getArg(index));
}

DagNode DagNode::getArgAsDagNode(unsigned index) const {
  return DagNode(dyn_cast_or_null<llvm::DagInit>(node->getArg(index)));
}

DagLeaf DagNode::getArgAsLeaf(unsigned index) const {
  assert(!isNestedDagArg(index));
  return DagLeaf(node->getArg(index));
}

StringRef DagNode::getArgName(unsigned index) const {
  return node->getArgNameStr(index);
}

bool DagNode::isUnspecified() const {
  return dyn_cast_or_null<llvm::UnsetInit>(node);
}

bool DagNode::isReplaceWithValue() const {
  auto *dagOpDef = cast<llvm::DefInit>(node->getOperator())->getDef();
  return dagOpDef->getName() == "replaceWithValue";
}

bool DagNode::isLocationDirective() const {
  auto *dagOpDef = cast<llvm::DefInit>(node->getOperator())->getDef();
  return dagOpDef->getName() == "location";
}

bool DagNode::isReturnTypeDirective() const {
  auto *dagOpDef = cast<llvm::DefInit>(node->getOperator())->getDef();
  return dagOpDef->getName() == "returnType";
}

bool DagNode::isEither() const {
  auto *dagOpDef = cast<llvm::DefInit>(node->getOperator())->getDef();
  return dagOpDef->getName() == "either";
}

bool DagNode::isRepeat() const {
  auto *dagOpDef = cast<llvm::DefInit>(node->getOperator())->getDef();
  return dagOpDef->getName() == "repeat";
}

bool DagNode::isYield() const {
  auto *dagOpDef = cast<llvm::DefInit>(node->getOperator())->getDef();
  return dagOpDef->getName() == "yield";
}


void DagNode::print(raw_ostream &os) const {
  if (node)
    node->print(os);
}

//===----------------------------------------------------------------------===//
// SymbolInfoMap
//===----------------------------------------------------------------------===//

StringRef SymbolInfoMap::getValuePackName(StringRef symbol, int *index) {
  StringRef name, indexStr;
  int idx = -1;
  std::tie(name, indexStr) = symbol.rsplit("__");

  if (indexStr.consumeInteger(10, idx)) {
    // The second part is not an index; we return the whole symbol as-is.
    return symbol;
  }
  if (index) {
    *index = idx;
  }
  return name;
}

SymbolInfoMap::SymbolInfo::SymbolInfo(const Operator *op, SymbolInfo::Kind kind,
                                      Optional<DagAndConstant> dagAndConstant)
    : op(op), kind(kind), dagAndConstant(std::move(dagAndConstant)) {}

int SymbolInfoMap::SymbolInfo::getStaticValueCount() const {
  switch (kind) {
  case Kind::Attr:
  case Kind::Operand:
  case Kind::Value:
  case Kind::TypeParam:
  case Kind::Yield:
    return 1;
  case Kind::Result:
    return op->getNumResults();
  case Kind::RepeatResult:
    //TODO(aviand): How many "static values" does a RepeatResult have?
    llvm::PrintFatalError("getStaticValueCount not yet implemented for RepeatResult.");
  case Kind::MultipleValues:
    return getSize();
  }
  llvm_unreachable("unknown kind");
}

std::string SymbolInfoMap::SymbolInfo::getVarName(StringRef name) const {
  return alternativeName ? *alternativeName : name.str();
}

bool SymbolInfoMap::SymbolInfo::isVariadic(StringRef name) const {
  switch (kind) {
  case Kind::Attr: {
    return false;
  }
  case Kind::Operand: {
    auto *operand = op->getArg(getArgIndex()).get<NamedTypeConstraint *>();
    return operand->isVariableLength();
  }
  case Kind::Result: {
    int index = -1;
    getValuePackName(name, &index);
    // If `index` is greater than zero, then we are referencing a specific
    // result of a multi-result op. The result can still be variadic.
    if (index >= 0)
      return op->getResult(index).isVariadic();

    return op->getNumResults() > 1;
  }
  case Kind::Value: {
    return false;
  }
  case Kind::MultipleValues: {
    return true;
  }
  case Kind::TypeParam: {
    return false;
  }
  case Kind::RepeatResult: {
    return true;
  }
  case Kind::Yield: {
    return false;
  }
  }
}

std::string SymbolInfoMap::SymbolInfo::getVarTypeStr(StringRef name) const {
  LLVM_DEBUG(llvm::dbgs() << "getVarTypeStr for '" << name << "': ");
  switch (kind) {
  case Kind::Attr: {
    if (op)
      return op->getArg(getArgIndex())
          .get<NamedAttribute *>()
          ->attr.getStorageType()
          .str();
    // TODO(suderman): Use a more exact type when available.
    return "::mlir::Attribute";
  }
  case Kind::Operand: {
    // Use operand range for captured operands (to support potential variadic
    // operands).
    return "::mlir::Operation::operand_range";
  }
  case Kind::Value: {
    return "::mlir::Value";
  }
  case Kind::MultipleValues: {
    return "::mlir::ValueRange";
  }
  case Kind::Result: {
    // Use the op itself for captured results.
    return op->getQualCppClassName();
  }
  case Kind::TypeParam: {
    //TODO(aviand): Remove 1000*opArgIndex + paramIndex hack
    const auto *opNode = (const llvm::DagInit*)  dagAndConstant->first;
    auto opArgIndex = dagAndConstant->second / 1000;
    auto * constraintNode =  (const llvm::DagInit*) opNode->getArg(opArgIndex);
    auto * dagOperator = constraintNode->getOperator();
    auto * defInit = llvm::cast_or_null<llvm::DefInit>(dagOperator);
    auto * record = defInit->getDef();
    auto attrOrTypeDef = AttrOrTypeDef(record);
    auto paramIndex = dagAndConstant->second % 1000;
    auto type = attrOrTypeDef.getParameters()[paramIndex].getCppType();
    return type.str();
  }
  case Kind::RepeatResult: {
    return "SmallVector<" + op->getQualCppClassName() + ", 4>";
  }
  case Kind::Yield: {
    return "::mlir::Value";
  }
  }
  llvm_unreachable("unknown kind");
}

std::string SymbolInfoMap::SymbolInfo::getTypeParamAccessor(StringRef name) const {
  switch (kind) {
  case Kind::TypeParam: {
    //TODO(aviand): Remove 1000*opArgIndex + paramIndex hack
    const auto *opNode = (const llvm::DagInit*)  dagAndConstant->first;
    auto opArgIndex = dagAndConstant->second / 1000;
    auto * constraintNode =  (const llvm::DagInit*) opNode->getArg(opArgIndex);
    auto *defInit =
        llvm::cast_or_null<llvm::DefInit>(constraintNode->getOperator());
    auto type = AttrOrTypeDef(defInit->getDef());
    auto paramIndex = dagAndConstant->second % 1000;
    auto parm = type.getParameters()[paramIndex];
    auto getParameterAccessorName = [](StringRef name) {
      auto ret = "get" + name.str();
      ret[3] = llvm::toUpper(ret[3]); // uppercase first letter of the name
      return ret;
    };
    return ".getType().cast<" +  type.getCppClassName().str() + ">()." +
           getParameterAccessorName(parm.getName()) + "()";
  }
  case Kind::Attr:
  case Kind::Operand:
  case Kind::Value:
  case Kind::MultipleValues:
  case Kind::Result:
  case Kind::RepeatResult:
  case Kind::Yield:
    assert(kind == Kind::TypeParam &&
           "Cannot call getTypeParamAccessor on other kinds");
  }
  llvm_unreachable("unknown kind");
}

std::string SymbolInfoMap::SymbolInfo::getVarDecl(StringRef name) const {
  LLVM_DEBUG(llvm::dbgs() << "getVarDecl for '" << name << "': ");
  std::string varInit = kind == Kind::Operand ? "(op0->getOperands())" : "";
  return std::string(
      formatv("{0} {1}{2};\n", getVarTypeStr(name), getVarName(name), varInit));
}

std::string SymbolInfoMap::SymbolInfo::getArgDecl(StringRef name) const {
  LLVM_DEBUG(llvm::dbgs() << "getArgDecl for '" << name << "': ");
  return std::string(
      formatv("{0} &{1}", getVarTypeStr(name), getVarName(name)));
}

std::string SymbolInfoMap::SymbolInfo::getValueAndRangeUse(
    StringRef name, int index, const char *fmt, const char *separator) const {
  LLVM_DEBUG(llvm::dbgs() << "getValueAndRangeUse for '" << name << "': ");
  switch (kind) {
  case Kind::Attr: {
    assert(index < 0);
    auto repl = formatv(fmt, name);
    LLVM_DEBUG(llvm::dbgs() << repl << " (Attr)\n");
    return std::string(repl);
  }
  case Kind::Operand: {
    assert(index < 0);
    auto *operand = op->getArg(getArgIndex()).get<NamedTypeConstraint *>();
    // If this operand is variadic, then return a range. Otherwise, return the
    // value itself.
    if (operand->isVariableLength()) {
      auto repl = formatv(fmt, name);
      LLVM_DEBUG(llvm::dbgs() << repl << " (VariadicOperand)\n");
      return std::string(repl);
    }
    auto repl = formatv(fmt, formatv("(*{0}.begin())", name));
    LLVM_DEBUG(llvm::dbgs() << repl << " (SingleOperand)\n");
    return std::string(repl);
  }
  case Kind::Result: {
    // If `index` is greater than zero, then we are referencing a specific
    // result of a multi-result op. The result can still be variadic.
    if (index >= 0) {
      std::string v =
          std::string(formatv("{0}.getODSResults({1})", name, index));
      if (!op->getResult(index).isVariadic())
        v = std::string(formatv("(*{0}.begin())", v));
      auto repl = formatv(fmt, v);
      LLVM_DEBUG(llvm::dbgs() << repl << " (SingleResult)\n");
      return std::string(repl);
    }

    // If this op has no result at all but still we bind a symbol to it, it
    // means we want to capture the op itself.
    if (op->getNumResults() == 0) {
      LLVM_DEBUG(llvm::dbgs() << name << " (Op)\n");
      return formatv(fmt, name);
    }

    // We are referencing all results of the multi-result op. A specific result
    // can either be a value or a range. Then join them with `separator`.
    SmallVector<std::string, 4> values;
    values.reserve(op->getNumResults());

    for (int i = 0, e = op->getNumResults(); i < e; ++i) {
      std::string v = std::string(formatv("{0}.getODSResults({1})", name, i));
      if (!op->getResult(i).isVariadic()) {
        v = std::string(formatv("(*{0}.begin())", v));
      }
      values.push_back(std::string(formatv(fmt, v)));
    }
    auto repl = llvm::join(values, separator);
    LLVM_DEBUG(llvm::dbgs() << repl << " (VariadicResult)\n");
    return repl;
  }
  case Kind::Value: {
    assert(index < 0);
    assert(op == nullptr);
    auto repl = formatv(fmt, name);
    LLVM_DEBUG(llvm::dbgs() << repl << " (Value)\n");
    return std::string(repl);
  }
  case Kind::MultipleValues: {
    assert(op == nullptr);
    assert(index < getSize());
    if (index >= 0) {
      std::string repl =
          formatv(fmt, std::string(formatv("{0}[{1}]", name, index)));
      LLVM_DEBUG(llvm::dbgs() << repl << " (MultipleValues)\n");
      return repl;
    }
    // If it doesn't specify certain element, unpack them all.
    auto repl =
        formatv(fmt, std::string(formatv("{0}.begin(), {0}.end()", name)));
    LLVM_DEBUG(llvm::dbgs() << repl << " (MultipleValues)\n");
    return std::string(repl);
  }
  case Kind::TypeParam: {
    //TODO(aviand): Support TypeParam in getValueAndRangeUse
    auto repl = formatv(fmt, name);
    LLVM_DEBUG(llvm::dbgs() << repl << " (TypeParam)\n");
    return std::string(repl);
  }
  case Kind::RepeatResult: {

    // Check if we're being asked for <var> or <var>__i
    // The latter indicates we're inside the repeat where <var> is matched
    StringRef outerVar, suffix;
    std::tie(outerVar, suffix) = StringRef(name).rsplit("__");
    bool insidePattern = suffix == "i";

    // If `index` is greater than zero, then we are referencing a specific
    // "iteration" version of this result
    if (index >= 0) {
      std::string v = std::string(formatv("{0}[{1}]", name, index));
      if (!op->getResult(index).isVariadic())
        v = std::string(formatv("(*{0}.begin())", v));
      auto repl = formatv(fmt, v);
      LLVM_DEBUG(llvm::dbgs() << repl << " (SingleRepeatResult)\n");
      return std::string(repl);
    }

    // TODO(aviand): Is capturing op itself necessary for repeat?

    // // If this op has no result at all but still we bind a symbol to it, it
    // // means we want to capture the op itself.
    // if (op->getNumResults() == 0) {
    //   LLVM_DEBUG(llvm::dbgs() << name << " (Op)\n");
    //   return std::string(name);
    // }

    // We are referencing all results of the repeated op. A specific result
    // can either be a value or a range. Then join them with `separator`.
    SmallVector<std::string, 4> values;
    values.reserve(op->getNumResults());

    if (insidePattern) {
      for (int i = 0, e = op->getNumResults(); i < e; ++i) {
        std::string v = std::string(
            formatv("{0}.getODSResults({1})", name, i));
        if (!op->getResult(i).isVariadic()) {
          v = std::string(formatv("(*{0}.begin())", v));
        }
        values.push_back(std::string(formatv(fmt, v)));
      }
      auto repl = llvm::join(values, separator);
      LLVM_DEBUG(llvm::dbgs() << repl << " (EntireRepeatResult inside Repeat)\n");
      return repl;
    }
    // TODO(aviand): properly support multi-result ops outside repeat pattern
    LLVM_DEBUG(llvm::dbgs() << outerVar << " (EntireRepeatResult outside Repeat)\n");
    return outerVar.str();
  }
  case Kind::Yield: {
    //TODO(aviand): Support Yield in getValueAndRangeUse
    auto repl = formatv(fmt, name);
    LLVM_DEBUG(llvm::dbgs() << repl << " (Yield)\n");
    return std::string(repl);
  }
  }
  llvm_unreachable("unknown kind");
}

std::string SymbolInfoMap::SymbolInfo::getAllRangeUse(
    StringRef name, int index, const char *fmt, const char *separator) const {
  LLVM_DEBUG(llvm::dbgs() << "getAllRangeUse for '" << name << "': ");
  switch (kind) {
  case Kind::Attr:
  case Kind::Operand: {
    assert(index < 0 && "only allowed for symbol bound to result");
    auto repl = formatv(fmt, name);
    LLVM_DEBUG(llvm::dbgs() << repl << " (Operand/Attr)\n");
    return std::string(repl);
  }
  case Kind::Result: {
    if (index >= 0) {
      auto repl = formatv(fmt, formatv("{0}.getODSResults({1})", name, index));
      LLVM_DEBUG(llvm::dbgs() << repl << " (SingleResult)\n");
      return std::string(repl);
    }

    // We are referencing all results of the multi-result op. Each result should
    // have a value range, and then join them with `separator`.
    SmallVector<std::string, 4> values;
    values.reserve(op->getNumResults());

    for (int i = 0, e = op->getNumResults(); i < e; ++i) {
      values.push_back(std::string(
          formatv(fmt, formatv("{0}.getODSResults({1})", name, i))));
    }
    auto repl = llvm::join(values, separator);
    LLVM_DEBUG(llvm::dbgs() << repl << " (VariadicResult)\n");
    return repl;
  }
  case Kind::Value: {
    assert(index < 0 && "only allowed for symbol bound to result");
    assert(op == nullptr);
    auto repl = formatv(fmt, formatv("{{{0}}", name));
    LLVM_DEBUG(llvm::dbgs() << repl << " (Value)\n");
    return std::string(repl);
  }
  case Kind::MultipleValues: {
    assert(op == nullptr);
    assert(index < getSize());
    if (index >= 0) {
      std::string repl =
          formatv(fmt, std::string(formatv("{0}[{1}]", name, index)));
      LLVM_DEBUG(llvm::dbgs() << repl << " (MultipleValues)\n");
      return repl;
    }
    auto repl =
        formatv(fmt, std::string(formatv("{0}.begin(), {0}.end()", name)));
    LLVM_DEBUG(llvm::dbgs() << repl << " (MultipleValues)\n");
    return std::string(repl);
  }
  case Kind::TypeParam: {
    //TODO(aviand): What to assert for TypeParam in getAllRangeUse?
    auto repl = formatv(fmt, formatv("{{{0}}", name));
    LLVM_DEBUG(llvm::dbgs() << repl << " (TypeRange)\n");
    return std::string(repl);
  }
  case Kind::RepeatResult: {
    if (index >= 0) {
      auto repl = formatv(fmt, formatv("{0}[{1}]", name, index));
      LLVM_DEBUG(llvm::dbgs() << repl << " (SingleRepeatResult)\n");
      return std::string(repl);
    }

    // We are referencing all results of the multi-result op. Each result should
    // have a value range, and then join them with `separator`.
    SmallVector<std::string, 4> values;
    values.reserve(op->getNumResults());

    for (int i = 0, e = op->getNumResults(); i < e; ++i) {
      values.push_back(std::string(
          formatv(fmt, formatv("{0}.WOOPPATTERNCPP626({1})", name, i))));
    }
    auto repl = llvm::join(values, separator);
    LLVM_DEBUG(llvm::dbgs() << repl << " (EntireRepeatResult)\n");
    return repl;
  }
  case Kind::Yield: {
    //TODO(aviand): What to assert for Yield in getAllRangeUse?
    auto repl = formatv(fmt, formatv("{{{0}}", name));
    LLVM_DEBUG(llvm::dbgs() << repl << " (Yield)\n");
    return std::string(repl);
  }
  }
  llvm_unreachable("unknown kind");
}

bool SymbolInfoMap::SymbolInfo::isTypeParam() const {
  return kind == Kind::TypeParam;
}

bool SymbolInfoMap::SymbolInfo::isAttribute() const {
  return kind == Kind::Attr;
}

bool SymbolInfoMap::SymbolInfo::isValue() const {
  return kind == Kind::Value;
}

bool SymbolInfoMap::SymbolInfo::isRepeatResult() const {
  return kind == Kind::RepeatResult;
}

bool SymbolInfoMap::bindOpArgument(DagNode node, StringRef symbol,
                                   const Operator &op, int argIndex) {
  assert(!symbol.empty());
  StringRef name = getValuePackName(symbol);
  if (name != symbol) {
    auto error = formatv(
        "symbol '{0}' with trailing index cannot bind to op argument", symbol);
    PrintFatalError(loc, error);
  }

  auto symInfo = op.getArg(argIndex).is<NamedAttribute *>()
                     ? SymbolInfo::getAttr(&op, argIndex)
                     : SymbolInfo::getOperand(node, &op, argIndex);

  std::string key = symbol.str();
  if (symbolInfoMap.count(key)) {
    // Only non unique name for the operand is supported.
    if (symInfo.kind != SymbolInfo::Kind::Operand) {
      return false;
    }

    // Cannot add new operand if there is already non operand with the same
    // name.
    if (symbolInfoMap.find(key)->second.kind != SymbolInfo::Kind::Operand) {
      return false;
    }
  }

  symbolInfoMap.emplace(key, symInfo);
  return true;
}

bool SymbolInfoMap::bindTypeArgument(DagNode opNode, const Operator &op, int opArgIndex, StringRef symbol,
                                     int typeParamIndex) {
  assert(!symbol.empty());
  LLVM_DEBUG(llvm::dbgs() << "binding type argument " << symbol
                          << " (op dag: " << opNode.node
                          << ", arg index: " << opArgIndex
                          << ", param index: " << typeParamIndex << ")\n");
  StringRef name = getValuePackName(symbol);
  if (name != symbol) {
    auto error = formatv(
        "symbol '{0}' with trailing index cannot bind to type argument", symbol);
    PrintFatalError(loc, error);
  }

  auto symInfo = SymbolInfo::getTypeParam(&op, opNode, opArgIndex,
                                          opNode.getArgAsDagNode(opArgIndex),
                                          typeParamIndex);

  std::string key = symbol.str();
  if (symbolInfoMap.count(key)) {
    // Only non unique name for the operand is supported.
    if (symInfo.kind != SymbolInfo::Kind::TypeParam) {
      return false;
    }

    // Cannot add new operand if there is already non operand with the same
    // name.
    if (symbolInfoMap.find(key)->second.kind != SymbolInfo::Kind::TypeParam) {
      return false;
    }
  }

  symbolInfoMap.emplace(key, symInfo);
  return true;
}

bool SymbolInfoMap::bindOpResult(StringRef symbol, const Operator &op) {
  assert(!symbol.empty());
  std::string name = getValuePackName(symbol).str();
  auto inserted = symbolInfoMap.emplace(name, SymbolInfo::getResult(&op));

  return symbolInfoMap.count(inserted->first) == 1;
}

bool SymbolInfoMap::bindOpRepeatResult(StringRef symbol, const Operator &op) {
  assert(!symbol.empty());
  std::string name = getValuePackName(symbol).str();
  auto inserted = symbolInfoMap.emplace(name, SymbolInfo::getRepeatResult(&op));

  return isBoundAlready(inserted->first);
}

bool SymbolInfoMap::bindValues(StringRef symbol, int numValues) {
  assert(!symbol.empty());
  std::string name = getValuePackName(symbol).str();
  if (numValues > 1)
    return bindMultipleValues(name, numValues);
  return bindValue(name);
}


bool SymbolInfoMap::isBoundAlready(std::string name) const {
  int count = 0;
  for(auto &p : symbolInfoMap) {
    count += p.first == name && p.second.kind != SymbolInfo::Kind::Yield;
  }
  return count == 1;
}

bool SymbolInfoMap::bindValue(StringRef symbol) {
  assert(!symbol.empty());
  auto inserted = symbolInfoMap.emplace(symbol.str(), SymbolInfo::getValue());
  return isBoundAlready(inserted->first);
}

bool SymbolInfoMap::bindYield(StringRef symbol) {
  assert(!symbol.empty());
  auto inserted = symbolInfoMap.emplace(symbol.str(), SymbolInfo::getYield());
  // Since this is a Yield, we can't use isBoundAlready, which ignores Yield symbols
  return symbolInfoMap.count(inserted->first) == 1;
}

bool SymbolInfoMap::bindMultipleValues(StringRef symbol, int numValues) {
  assert(!symbol.empty());
  std::string name = getValuePackName(symbol).str();
  auto inserted =
      symbolInfoMap.emplace(name, SymbolInfo::getMultipleValues(numValues));
  return isBoundAlready(inserted->first);
}

bool SymbolInfoMap::bindAttr(StringRef symbol) {
  assert(!symbol.empty());
  auto inserted = symbolInfoMap.emplace(symbol.str(), SymbolInfo::getAttr());
  return isBoundAlready(inserted->first);
}

bool SymbolInfoMap::contains(StringRef symbol) const {
  return find(symbol) != symbolInfoMap.end();
}

SymbolInfoMap::const_iterator SymbolInfoMap::find(StringRef key) const {
  std::string name = getValuePackName(key).str();

  return symbolInfoMap.find(name);
}

SymbolInfoMap::const_iterator
SymbolInfoMap::findBoundSymbol(StringRef key, DagNode node, const Operator &op,
                               int argIndex) const {
  return findBoundSymbol(key, SymbolInfo::getOperand(node, &op, argIndex));
}

SymbolInfoMap::const_iterator
SymbolInfoMap::findBoundSymbol(StringRef key, DagNode opNode, const Operator &op, int opArgIndex,
                               DagNode constraintNode, int paramIndex) const {
  return findBoundSymbol(key,
                         SymbolInfo::getTypeParam(&op, opNode, opArgIndex,
                                                  constraintNode, paramIndex));
}

SymbolInfoMap::const_iterator
SymbolInfoMap::findBoundSymbol(StringRef key,
                               const SymbolInfo &symbolInfo) const {
  std::string name = getValuePackName(key).str();
  auto range = symbolInfoMap.equal_range(name);

  //TODO(aviand): Remove 1000*opArgIndex + paramIndex hack
  for (auto it = range.first; it != range.second; ++it)
    if (it->second.isTypeParam()) {
      if (it->second.dagAndConstant->first == symbolInfo.dagAndConstant->first)
        if (it->second.dagAndConstant->second / 1000 ==
            symbolInfo.dagAndConstant->second / 1000)
          return it;
    } else {
      if (it->second.dagAndConstant == symbolInfo.dagAndConstant)
        return it;
    }

  return symbolInfoMap.end();
}

std::pair<SymbolInfoMap::iterator, SymbolInfoMap::iterator>
SymbolInfoMap::getRangeOfEqualElements(StringRef key) {
  std::string name = getValuePackName(key).str();

  return symbolInfoMap.equal_range(name);
}

int SymbolInfoMap::count(StringRef key) const {
  std::string name = getValuePackName(key).str();
  return symbolInfoMap.count(name);
}

int SymbolInfoMap::getStaticValueCount(StringRef symbol) const {
  StringRef name = getValuePackName(symbol);
  if (name != symbol) {
    // If there is a trailing index inside symbol, it references just one
    // static value.
    return 1;
  }
  // Otherwise, find how many it represents by querying the symbol's info.
  return find(name)->second.getStaticValueCount();
}

std::string SymbolInfoMap::getValueAndRangeUse(StringRef symbol,
                                               const char *fmt,
                                               const char *separator) const {

  // Remove the pattern suffix if it is present
  bool hadSuffix = symbol.endswith("__i");
  if (hadSuffix) {
    symbol = symbol.substr(0, symbol.size() - 3);
  }

  int index = -1;
  StringRef name = getValuePackName(symbol, &index);

  auto it = symbolInfoMap.find(name.str());
  if (it == symbolInfoMap.end()) {
    auto error = formatv("referencing unbound symbol '{0}'", symbol);
    PrintFatalError(loc, error);
  }


  if (hadSuffix)
    name = StringRef(name.str() + "__i");
  return it->second.getValueAndRangeUse(name, index, fmt, separator);
}

std::string SymbolInfoMap::getAllRangeUse(StringRef symbol, const char *fmt,
                                          const char *separator) const {
  int index = -1;
  StringRef name = getValuePackName(symbol, &index);

  auto it = symbolInfoMap.find(name.str());
  if (it == symbolInfoMap.end()) {
    auto error = formatv("referencing unbound symbol '{0}'", symbol);
    PrintFatalError(loc, error);
  }

  return it->second.getAllRangeUse(name, index, fmt, separator);
}

void SymbolInfoMap::assignUniqueAlternativeNames() {
  llvm::StringSet<> usedNames;

  for (auto symbolInfoIt = symbolInfoMap.begin();
       symbolInfoIt != symbolInfoMap.end();) {
    auto range = symbolInfoMap.equal_range(symbolInfoIt->first);
    auto startRange = range.first;
    auto endRange = range.second;

    auto operandName = symbolInfoIt->first;
    int startSearchIndex = 0;
    for (++startRange; startRange != endRange; ++startRange) {
      // Current operand name is not unique, find a unique one
      // and set the alternative name.
      for (int i = startSearchIndex;; ++i) {
        std::string alternativeName = operandName + std::to_string(i);
        if (!usedNames.contains(alternativeName) &&
            symbolInfoMap.count(alternativeName) == 0) {
          usedNames.insert(alternativeName);
          startRange->second.alternativeName = alternativeName;
          startSearchIndex = i + 1;

          break;
        }
      }
    }

    symbolInfoIt = endRange;
  }
}

//===----------------------------------------------------------------------===//
// Pattern
//==----------------------------------------------------------------------===//

Pattern::Pattern(const llvm::Record *def, RecordOperatorMap *mapper)
    : def(*def), recordOpMap(mapper) {}

DagNode Pattern::getSourcePattern() const {
  return DagNode(def.getValueAsDag("sourcePattern"));
}

int Pattern::getNumResultPatterns() const {
  auto *results = def.getValueAsListInit("resultPatterns");
  return results->size();
}

DagNode Pattern::getResultPattern(unsigned index) const {
  auto *results = def.getValueAsListInit("resultPatterns");
  return DagNode(cast<llvm::DagInit>(results->getElement(index)));
}

void Pattern::collectSourcePatternBoundSymbols(SymbolInfoMap &infoMap) {
  LLVM_DEBUG(llvm::dbgs() << "start collecting source pattern bound symbols\n");
  collectBoundSymbols(getSourcePattern(), infoMap, /*isSrcPattern=*/true);
  LLVM_DEBUG(llvm::dbgs() << "done collecting source pattern bound symbols\n");

  LLVM_DEBUG(llvm::dbgs() << "start assigning alternative names for symbols\n");
  infoMap.assignUniqueAlternativeNames();
  LLVM_DEBUG(llvm::dbgs() << "done assigning alternative names for symbols\n");
}

void Pattern::collectResultPatternBoundSymbols(SymbolInfoMap &infoMap) {
  LLVM_DEBUG(llvm::dbgs() << "start collecting result pattern bound symbols\n");
  for (int i = 0, e = getNumResultPatterns(); i < e; ++i) {
    auto pattern = getResultPattern(i);
    collectBoundSymbols(pattern, infoMap, /*isSrcPattern=*/false);
  }
  LLVM_DEBUG(llvm::dbgs() << "done collecting result pattern bound symbols\n");
}

const Operator &Pattern::getSourceRootOp() {
  return getSourcePattern().getDialectOp(recordOpMap);
}

Operator &Pattern::getDialectOp(DagNode node) {
  return node.getDialectOp(recordOpMap);
}

std::vector<AppliedConstraint> Pattern::getConstraints() const {
  auto *listInit = def.getValueAsListInit("constraints");
  std::vector<AppliedConstraint> ret;
  ret.reserve(listInit->size());

  for (auto *it : *listInit) {
    auto *dagInit = dyn_cast<llvm::DagInit>(it);
    if (!dagInit)
      PrintFatalError(&def, "all elements in Pattern multi-entity "
                            "constraints should be DAG nodes");

    std::vector<std::string> entities;
    entities.reserve(dagInit->arg_size());
    for (auto *argName : dagInit->getArgNames()) {
      if (!argName) {
        PrintFatalError(
            &def,
            "operands to additional constraints can only be symbol references");
      }
      entities.emplace_back(argName->getValue());
    }

    ret.emplace_back(cast<llvm::DefInit>(dagInit->getOperator())->getDef(),
                     dagInit->getNameStr(), std::move(entities));
  }
  return ret;
}

int Pattern::getBenefit() const {
  // The initial benefit value is a heuristic with number of ops in the source
  // pattern.
  int initBenefit = getSourcePattern().getNumOps();
  llvm::DagInit *delta = def.getValueAsDag("benefitDelta");
  if (delta->getNumArgs() != 1 || !isa<llvm::IntInit>(delta->getArg(0))) {
    PrintFatalError(&def,
                    "The 'addBenefit' takes and only takes one integer value");
  }
  return initBenefit + dyn_cast<llvm::IntInit>(delta->getArg(0))->getValue();
}

std::vector<Pattern::IdentifierLine> Pattern::getLocation() const {
  std::vector<std::pair<StringRef, unsigned>> result;
  result.reserve(def.getLoc().size());
  for (auto loc : def.getLoc()) {
    unsigned buf = llvm::SrcMgr.FindBufferContainingLoc(loc);
    assert(buf && "invalid source location");
    result.emplace_back(
        llvm::SrcMgr.getBufferInfo(buf).Buffer->getBufferIdentifier(),
        llvm::SrcMgr.getLineAndColumn(loc, buf).first);
  }
  return result;
}

void Pattern::verifyBind(bool result, StringRef symbolName) {
  if (!result) {
    auto err = formatv("symbol '{0}' bound more than once", symbolName);
    PrintFatalError(&def, err);
  }
}

void Pattern::collectBoundSymbols(DagNode tree, SymbolInfoMap &infoMap,
                                  bool isSrcPattern, bool isRepeatPattern) {
  auto treeName = tree.getSymbol();
  auto numTreeArgs = tree.getNumArgs();

  if (tree.isNativeCodeCall()) {
    if (!treeName.empty()) {
      if (!isSrcPattern) {
        LLVM_DEBUG(llvm::dbgs() << "found symbol bound to NativeCodeCall: "
                                << treeName << '\n');
        verifyBind(
            infoMap.bindValues(treeName, tree.getNumReturnsOfNativeCode()),
            treeName);
      } else {
        PrintFatalError(&def,
                        formatv("binding symbol '{0}' to NativecodeCall in "
                                "MatchPattern is not supported",
                                treeName));
      }
    }

    for (int i = 0; i != numTreeArgs; ++i) {
      if (auto treeArg = tree.getArgAsDagNode(i)) {
        // This DAG node argument is a DAG node itself. Go inside recursively.
        collectBoundSymbols(treeArg, infoMap, isSrcPattern);
        continue;
      }

      if (!isSrcPattern)
        continue;

      // We can only bind symbols to arguments in source pattern. Those
      // symbols are referenced in result patterns.
      auto treeArgName = tree.getArgName(i);

      // `$_` is a special symbol meaning ignore the current argument.
      if (!treeArgName.empty() && treeArgName != "_") {
        DagLeaf leaf = tree.getArgAsLeaf(i);

        // In (NativeCodeCall<"Foo($_self, $0, $1, $2)"> I8Attr:$a, I8:$b, $c),
        if (leaf.isUnspecified()) {
          // This is case of $c, a Value without any constraints.
          verifyBind(infoMap.bindValue(treeArgName), treeArgName);
        } else {
          auto constraint = leaf.getAsConstraint();
          bool isAttr = leaf.isAttrConstraint() || leaf.isEnumAttrCase() ||
                        leaf.isConstantAttr() ||
                        constraint.getKind() == Constraint::Kind::CK_Attr;

          if (isAttr) {
            // This is case of $a, a binding to a certain attribute.
            verifyBind(infoMap.bindAttr(treeArgName), treeArgName);
            continue;
          }

          // This is case of $b, a binding to a certain type.
          verifyBind(infoMap.bindValue(treeArgName), treeArgName);
        }
      }
    }

    return;
  }

  if (tree.isRepeat()) {
    if (!treeName.empty()) {
      PrintFatalError(
          &def, formatv("binding symbol to Repeat is not supported", treeName));
    }

    auto indexName = tree.getArgName(2);
    auto index =  tree.getArgAsLeaf(2);
    if(index.isUnspecified()) {
      verifyBind(infoMap.bindValue(indexName), indexName);
    } else {
      llvm::PrintFatalError(formatv("Repeat's index ({0}) must be "
                                    "unspecified (?)",
                                    indexName));
    }

    auto yield = tree.getArgAsDagNode(3);
    int startArgIndex = 3;
    if (yield.isYield()) {
      auto yieldName = yield.getSymbol();
      verifyBind(infoMap.bindYield(yieldName), yieldName);
      startArgIndex++;
    }

    for (int i = startArgIndex; i != numTreeArgs; ++i) {
      if (auto treeArg = tree.getArgAsDagNode(i)) {
        // This DAG node argument is a DAG node itself. Go inside recursively.
        collectBoundSymbols(treeArg, infoMap, isSrcPattern, true);
      } else {
        llvm::PrintFatalError(
            formatv("Repeat argument {0} must be a nested DAG", i));
      }
    }
    return;
  }

  if (tree.isOperation()) {
    auto &op = getDialectOp(tree);
    auto numOpArgs = op.getNumArgs();
    int numEither = 0;

    // We need to exclude the trailing directives and `either` directive groups
    // two operands of the operation.
    int numDirectives = 0;
    for (int i = numTreeArgs - 1; i >= 0; --i) {
      if (auto dagArg = tree.getArgAsDagNode(i)) {
        if (dagArg.isLocationDirective() || dagArg.isReturnTypeDirective())
          ++numDirectives;
        else if (dagArg.isEither())
          ++numEither;
      }
    }
    auto numArgs = numTreeArgs - numDirectives + numEither;
    auto numVariadicArgs = op.getNumVariableLengthOperands();
    if (numArgs < numOpArgs || (numVariadicArgs != 1 && numArgs > numOpArgs)) {
      auto err =
          formatv("op '{0}' argument number mismatch: "
                  "{1} in pattern vs. {2} in definition",
                  op.getOperationName(), numTreeArgs + numEither, numOpArgs);
      PrintFatalError(&def, err);
    }

    // The name attached to the DAG node's operator is for representing the
    // results generated from this op. It should be remembered as bound results.
    if (!treeName.empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "found symbol bound to op result: " << treeName << '\n');
      if(isRepeatPattern)
        verifyBind(infoMap.bindOpRepeatResult(treeName, op), treeName);
      else
      verifyBind(infoMap.bindOpResult(treeName, op), treeName);
    }

    // The operand in `either` DAG should be bound to the operation in the
    // parent DagNode.
    auto collectSymbolInEither = [&](DagNode parent, DagNode tree,
                                     int &opArgIdx) {
      //TODO(aviand): Support Either appearing inside a repeat pattern
      for (int i = 0; i < tree.getNumArgs(); ++i, ++opArgIdx) {
        if (DagNode subTree = tree.getArgAsDagNode(i)) {
          collectBoundSymbols(subTree, infoMap, isSrcPattern);
        } else {
          auto argName = tree.getArgName(i);
          if (!argName.empty() && argName != "_")
            verifyBind(infoMap.bindOpArgument(parent, argName, op, opArgIdx),
                       argName);
        }
      }
    };

    for (int i = 0, opArgIdx = 0; i != numTreeArgs; ++i, ++opArgIdx) {
      if (auto treeArg = tree.getArgAsDagNode(i)) {
        if (treeArg.isEither()) {
          collectSymbolInEither(tree, treeArg, opArgIdx);
        } else if (treeArg.isComplexTypeConstraint()) {
          // Bind main value name
          auto operandName = tree.getArgName(opArgIdx);
          verifyBind(infoMap.bindOpArgument(tree, operandName, op, opArgIdx),
                     operandName);

          // ComplexTypeConstraint
          for (int j = 0; j < treeArg.getNumArgs(); ++j) {
            auto typeParamName = treeArg.getArgName(j);
            if (!typeParamName.empty() && typeParamName != "_")
              verifyBind(infoMap.bindTypeArgument(tree, op, opArgIdx,
                                                  typeParamName, j),
                         typeParamName);
          }

        } else {
          // This DAG node argument is a DAG node itself. Go inside recursively.
          collectBoundSymbols(treeArg, infoMap, isSrcPattern);
        }
        continue;
      }

      if (isSrcPattern) {
        // We can only bind symbols to op arguments in source pattern. Those
        // symbols are referenced in result patterns.
        auto treeArgName = tree.getArgName(i);
        // `$_` is a special symbol meaning ignore the current argument.
        if (!treeArgName.empty() && treeArgName != "_") {
          LLVM_DEBUG(llvm::dbgs() << "found symbol bound to op argument: "
                                  << treeArgName << '\n');
          verifyBind(infoMap.bindOpArgument(tree, treeArgName, op, opArgIdx),
                     treeArgName);
        }
      }
    }
    return;
  }

  if (!treeName.empty()) {
    PrintFatalError(
        &def, formatv("binding symbol '{0}' to non-operation/native code call "
                      "unsupported right now",
                      treeName));
  }
}
