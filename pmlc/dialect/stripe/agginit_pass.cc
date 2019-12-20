// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/transcode.h"
#include "pmlc/dialect/stripe/agginit_pass.h"

#include <sstream>
#include <vector>

#include "base/util/logging.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Translation.h"

#include "pmlc/dialect/stripe/analysis.h"
#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/ops.h"
#include "pmlc/dialect/stripe/transforms.h"
#include "pmlc/dialect/stripe/util.h"

namespace pmlc {
namespace dialect {
namespace stripe {

using vertexai::tile::codegen::proto::MLIR_AggInitPass;

class AggregateInitializer {
public:
  explicit AggregateInitializer(const MLIR_AggInitPass& opts);
  // Initialize the aggregate buffers in op
  void Initialize(ParallelForOp op);
private:
  // Find out where to insert the initialization of tensor
  std::pair<Operation*, Value*> InitLocation(Value* tensor);
  // Insert the initialization for the tensor in the aggregate op
  void InsertInit(AggregateOp aop);

  // Optimization options
  const MLIR_AggInitPass& options;
  // The current ParallelForOp
  ParallelForOp curOp;
  // The current op builder
  OpBuilder* builder;
};

AggregateInitializer::AggregateInitializer(const MLIR_AggInitPass& opts) : options(opts) {
}

// Find out where to insert the initialization of tensor
std::pair<Operation*, Value*> AggregateInitializer::InitLocation(Value* tensor) {
  Value* prev_tensor = nullptr;
  Operation* last_refine = nullptr;
  Value* cur_tensor = tensor;
  while (cur_tensor) {
    if (auto def_op = cur_tensor->getDefiningOp()) {
      if (auto allocate_op = mlir::dyn_cast<AllocateOp>(def_op)) {
        if (!last_refine) {
          throw std::runtime_error("Using tensor without refine op.");
        }
        // The initialization should be right after the allocation/refinement pair
        auto iter = last_refine->getIterator();
        ++iter;
        return std::make_pair(&(*iter), prev_tensor);
      }
      if (auto refine_op = mlir::dyn_cast<RefineOp>(def_op)) {
        // If refine_op is out of curOp, stop here and insert
        // the initialization right before curOp
        if (refine_op.getParentOp() == curOp.getParentOp()) {
          return std::make_pair(curOp, cur_tensor);
        }
        prev_tensor = cur_tensor;
        last_refine = refine_op.getOperation();
        cur_tensor = refine_op.in();
      }
    }
    else {
      throw std::runtime_error("Cannot find the tensor definition.");
    }
  }
  throw std::runtime_error("Cannot find the tensor definition.");
}

// Insert the initialization for the tensor in the aggregate op
void AggregateInitializer::InsertInit(AggregateOp aop) {
  Value* tensor = aop.into();
  llvm::SmallVector<AffineRange, 8> ranges;
  FlatTensorAccess flat_access = ComputeAccess(tensor);
  for (AffinePolynomial& ap : flat_access.access) {
    ranges.push_back(AffineRange(ap));
  }
  IVLOG(3, "Initialize " << tensorName(tensor).str());
  for (AffineRange& ap : ranges) {
    IVLOG(3, "    " << ap.min << " " << ap.max);
  }

  // Insert the initialization right before init_loc
  Operation* init_loc;
  Value* init_tensor;
  std::tie(init_loc, init_tensor) = InitLocation(tensor);

  // Collect loop ranges
  std::vector<int64_t> range_nums(ranges.size());
  llvm::SmallVector<Attribute, 8> idx_names(ranges.size());
  for (unsigned i = 0; i < ranges.size(); ++i) {
    const AffineRange& ar = ranges[i];
    range_nums[i] = ar.max - ar.min + 1;
    idx_names[i] = builder->getStringAttr("i" + std::to_string(i));
  }

  Location unknownLoc = builder->getUnknownLoc();
  builder->setInsertionPoint(init_loc);

  // Create loop body
  ParallelForOp loop_op = builder->create<ParallelForOp>(unknownLoc, builder->getI64ArrayAttr(range_nums));
  loop_op.setAttr("name", builder->getStringAttr("Init(" + tensorName(init_tensor).str() + ")"));
  loop_op.setAttr("comments", builder->getStringAttr(""));
  loop_op.setAttr("idx_names", ArrayAttr::get(idx_names, builder->getContext()));

  Block* loop_body = new Block();
  builder->setInsertionPointToStart(loop_body);

  // Determine the initial value
  DataType init_type = tensorElementType(init_tensor);
  eltwise::ScalarConstantOp const_op = initialValue(builder,
    init_type, util::stringifyAggregationKind(aop.agg()), "cst");

  // Create statements
  std::vector<Value*> offsets(ranges.size());
  for (unsigned i = 0; i < ranges.size(); ++i) {
    auto idx = loop_body->addArgument(AffineType::get(builder->getContext()));
    const AffineRange& ar = ranges[i];
    llvm::SmallVector<Value*, 1> vars = {idx};
    llvm::SmallVector<int64_t, 1> coeffs = {1};
    offsets[i] = builder->create<AffinePolyOp>(
      unknownLoc,
      builder->getType<AffineType>(),
      vars,
      builder->getI64ArrayAttr(coeffs),
      builder->getI64IntegerAttr(ar.min)
    );
  }
  auto ref_op = builder->create<RefineOp>(unknownLoc, init_tensor->getType(), init_tensor, offsets);
  builder->create<StoreOp>(unknownLoc, ref_op.result(), const_op.result());
  builder->create<TerminateOp>(unknownLoc);

  // Insert the loop body
  loop_op.getOperation()->getRegion(0).push_back(loop_body);

  // Parallelize the loop if it is top-level block
  if (options.parallel() && init_loc->getParentOp() == curOp.getParentOp()) {
    ParallelizeEltwise(loop_op, options.cache_line() / byte_width(init_type), "cpu_thread");
  }
}

// Initialize the aggregate buffers in op
void AggregateInitializer::Initialize(ParallelForOp op) {
  Block* obody = op.getBody();
  curOp = op;
  OpBuilder op_builder = op.getBodyBuilder();
  builder = &op_builder;
  obody->walk([&](AggregateOp aop) {
    InsertInit(aop);
  });
}

void AggInitPass::runOnFunction() {
  auto reqs = vertexai::tile::stripe::FromProto(options.reqs());
  mlir::FuncOp f = getFunction();
  AggregateInitializer ai(options);
  f.walk([&reqs, &ai] (ParallelForOp op) {
    if (hasAttrs(op.getOperation(), reqs)) {
      ai.Initialize(op);
    }
  });
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
