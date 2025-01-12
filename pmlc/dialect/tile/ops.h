// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

#include "pmlc/dialect/eltwise/ops.h"
#include "pmlc/dialect/eltwise/types.h"
#include "pmlc/dialect/tile/types.h"
#include "pmlc/util/enums.h"

namespace pmlc::dialect::tile {

using eltwise::DataType;
using eltwise::ScalarType;
using llvm::APInt;
using llvm::Optional;
using mlir::AffineMap;
using mlir::AffineMapAttr;
using mlir::ArrayAttr;
using mlir::ArrayRef;
using mlir::Attribute;
using mlir::BoolAttr;
using mlir::Builder;
using mlir::IndexType;
using mlir::IntegerAttr;
using mlir::IntegerSet;
using mlir::IntegerSetAttr;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::NamedAttribute;
using mlir::NoneType;
using mlir::Op;
using mlir::OpAsmOpInterface;
using mlir::OpAsmParser;
using mlir::OpAsmPrinter;
using mlir::OpAsmSetValueNameFn;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::OpFoldResult;
using mlir::OpInterface;
using mlir::OwningRewritePatternList;
using mlir::ParseResult;
using mlir::RankedTensorType;
using mlir::Region;
using mlir::ShapedType;
using mlir::StringAttr;
using mlir::StringRef;
using mlir::TupleType;
using mlir::Type;
using mlir::TypeAttr;
using mlir::UnitAttr;
using mlir::Value;
using mlir::ValueRange;
using mlir::VectorType;
using util::AggregationKind;
using util::CombinationKind;
using util::GenericBuilder;

namespace OpTrait = mlir::OpTrait;

#include "pmlc/dialect/tile/interfaces.h.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/tile/ops.h.inc"

}  // namespace pmlc::dialect::tile
