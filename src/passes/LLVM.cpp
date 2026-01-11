/*
 * Copyright 2022 WebAssembly Community Group participants
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//
// Tested with LLVM 15.
//
// Use
// #define BINARYEN_LLVM_DEBUG 1
// to add debugging and logging.

#include <llvm/ADT/APInt.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/ValueSymbolTable.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <ir/iteration.h>
#include <ir/module-utils.h>
#include <ir/table-utils.h>
#include <pass.h>
#include <pretty_printing.h>
#include <wasm-binary.h>
#include <wasm-builder.h>
#include <wasm-stack.h>
#include <wasm.h>

using namespace llvm;

#if BINARYEN_LLVM_DEBUG
#define DUMP(x) (x)->dump()
#define LOG(x) std::cerr << "LOG: " << (x) << '\n'
#else
#define DUMP(x) (void)(x)
#define LOG(x)
#endif
#define PANIC(x) wasm::Fatal() << (x)

struct LLVMPass;

namespace wasm {
struct VisitLogger : public OverriddenVisitor<VisitLogger, Value*> {
  VisitLogger() {}
  virtual ~VisitLogger() {};

#define DELEGATE(CLASS_TO_VISIT)                                               \
  virtual Value* visit##CLASS_TO_VISIT(CLASS_TO_VISIT* curr) {                 \
    PANIC("  !!!! Not implemented --> visit" #CLASS_TO_VISIT);                 \
    return nullptr;                                                            \
  };

#include "wasm-delegations.def"
};
} // namespace wasm

struct LLVMPass : public wasm::Pass {
  // Global state. Each LLVM pass instance creates a context and the other data
  // structures we will need. We also create a single module for the lifetime of
  // the pass. As we compile code, we modify the contents inside that module by
  // adding and removing functions.
  Triple triple;
  std::unique_ptr<llvm::LLVMContext> context;
  std::unique_ptr<llvm::Module> mod;
  std::unique_ptr<llvm::IRBuilder<>> builder;
  std::unique_ptr<TargetMachine> targetMachine;

  llvm::Type* i1;
  llvm::Type* i32;
  llvm::Type* i64;
  llvm::Type* f32;
  llvm::Type* f64;
  llvm::Type* v128;
  llvm::Type* v01d;
  llvm::Type* funcRef;
  llvm::Type* anyRef;

  llvm::Function* nullFunc;

  llvm::Function* currentFunc;
  llvm::Value* lastValue;
  llvm::StringMap<BasicBlock*> namedBlocks;
  llvm::StringMap<Function*> namedFunctions;

  typedef std::tuple<BasicBlock*, Value*> BranchWithValue;

  std::map<BasicBlock*, std::vector<BranchWithValue>> branchMap;
  struct Memory {
    GlobalVariable* ptr;
    uint64_t size;    // current memory size in bytes
    uint64_t initial; // initial size in page
    uint64_t max;     // max size in page
  };
  // By now one WASM module can only access one memory
  llvm::StringMap<Memory> namedMemories;

  struct IREmitter : public wasm::VisitLogger {
    LLVMPass& parent;
    IREmitter(LLVMPass& parent) : parent(parent) {}

    Value* visit(wasm::Expression* curr) {
      auto value = wasm::VisitLogger::visit(curr);
      if (value != nullptr) {
        parent.lastValue = value;
      } else {
        LOG("Visiting ");
        DUMP(curr);
        LOG("Warning: Invalid(?) return value: nullptr");
      }
      return parent.lastValue;
    }

    Value* visitBlock(wasm::Block* curr) override;
    Value* visitMemoryFill(wasm::MemoryFill* curr) override;
    Value* visitConst(wasm::Const* curr) override;
    Value* visitLocalGet(wasm::LocalGet* curr) override;
    Value* visitLocalSet(wasm::LocalSet* curr) override;
    Value* visitStore(wasm::Store* curr) override;
    Value* visitLoad(wasm::Load* curr) override;
    Value* visitCall(wasm::Call* curr) override;
    Value* visitBinary(wasm::Binary* curr) override;
    Value* visitUnary(wasm::Unary* curr) override;
    Value* visitGlobalGet(wasm::GlobalGet* curr) override;
    Value* visitGlobalSet(wasm::GlobalSet* curr) override;
    Value* visitSelect(wasm::Select* curr) override;
    Value* visitReturn(wasm::Return* curr) override;
    Value* visitBreak(wasm::Break* curr) override;
    Value* visitLoop(wasm::Loop* curr) override;
    Value* visitDrop(wasm::Drop* curr) override;
    Value* visitUnreachable(wasm::Unreachable* curr) override;
    Value* visitMemorySize(wasm::MemorySize* curr) override;
    Value* visitMemoryCopy(wasm::MemoryCopy* curr) override;
    Value* visitSwitch(wasm::Switch* curr) override;
    Value* visitCallIndirect(wasm::CallIndirect* curr) override;
    Value* visitRefFunc(wasm::RefFunc* curr) override;
    Value* visitIf(wasm::If* curr) override;
    Value* visitNop(wasm::Nop* curr) override;
    Value* visitMemoryGrow(wasm::MemoryGrow* curr) override;

  private:
    Value* assurePtrType(Value* val, Type* elmTy);
    Value* assureIntType(Value* val, int bits);
    Value* assureType(Value* val, Type* ty);
    Function* assureFunction(const std::string& fname,
                             const std::vector<Type*>& paramTypes);
    Value* assureBool(Value* condition);
    Value* rotl(Value* a, Value* b);
    Value* rotr(Value* a, Value* b);
    Value* eqz(Value* v);
    BasicBlock* createBlock(const std::string& name,
                            Function* parent = nullptr);
    void iterateBody(wasm::Expression* body);
    void appendInsertionBlock(BasicBlock* block);
    void appendInsertionBlock(const std::string& name);

    friend struct LLVMPass;
  };

  std::unique_ptr<IREmitter> emitter;

  std::ostream& output;

  LLVMPass() : output(std::cout) {};

  LLVMPass(std::ostream& ostream) : output(ostream) {};

  // Initialization of LLVM.
  void initLLVM() {
    static bool done = false;
    if (done) {
      return;
    }

    // Perhaps we could call only LLVMInitializeWebAssemblyTargetInfo() etc?
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmPrinters();
    InitializeAllAsmParsers();

    done = true;
  }

  // Initialize this Pass instance's global state.
  void initPassInstance() {
    triple = Triple("wasm32-unknown-unknown");

    context = std::make_unique<LLVMContext>();

    i1 = Type::getInt1Ty(*context);
    i32 = Type::getInt32Ty(*context);
    i64 = Type::getInt64Ty(*context);
    f32 = Type::getFloatTy(*context);
    f64 = Type::getDoubleTy(*context);
    v128 = Type::getInt128Ty(*context);
    v01d = Type::getVoidTy(*context);
    // funcRef for a generic function pointer type
    funcRef = PointerType::getUnqual(FunctionType::get(i32, false));
    anyRef = PointerType::getUnqual(*context);

    currentFunc = nullptr;
    lastValue = nullptr;
    namedBlocks.clear();
    branchMap.clear();

    std::string error;

    auto target = TargetRegistry::lookupTarget(triple.getTriple(), error);
    if (!target) {
      wasm::Fatal() << "can't find wasm target: " << error;
    }

    targetMachine = std::unique_ptr<TargetMachine>(
      target->createTargetMachine(triple.getTriple(), "generic", "", {}, {}));

    emitter = std::make_unique<IREmitter>(*this);
  }

  // Reset the state of our global LLVM module, removing current changes so that
  // it is ready for further work later. This removes any functions we added,
  // after which the module is empty.
  void resetLLVMModule() {
    auto& list = mod->getFunctionList();
    list.clear();
    namedFunctions.clear();
    namedBlocks.clear();
    branchMap.clear();
  }

  inline std::string memoryName(std::string name) { return "Memory|" + name; }
  inline std::string tableName(std::string name) { return "Table|" + name; }
  inline std::string dataName(std::string name) { return "Data|" + name; }

  void handleHeapType(wasm::HeapType type) {
    LOG("Define the non-basic type " + type.toString());
  }

  void visitMemory(wasm::Memory* curr) {
    LOG("visitMemory to define Memory");

    auto name = memoryName(curr->name.toString());
    uint64_t initial = curr->initial;
    if (initial == 0) {
      initial = 1;
    }

    uint64_t max = curr->hasMax() ? (uint64_t)curr->max : curr->kMaxSize32;

    LOG(" .    Name: " + name);
    LOG(" .    Size (pages): " + std::to_string(initial));

    uint64_t size = initial * curr->kPageSize;

    auto mem_ty = IntegerType::get(*context, 8);
    auto arrayTy = ArrayType::get(mem_ty, size);

    auto linkage = GlobalVariable::PrivateLinkage;
    Constant* initializer = nullptr;
    if (curr->imported()) {
      linkage = GlobalVariable::ExternalLinkage;
    } else {
      initializer = ConstantAggregateZero::get(arrayTy);
    }

    GlobalVariable* mem =
      new GlobalVariable(*mod, arrayTy, false, linkage, initializer, name);

    LOG(" .  Memory: ");

    namedMemories[name] = {mem, size, initial, max};
  }

  void visitTable(wasm::Table* curr) {
    auto name = tableName(curr->name.toString());
    LOG("Define table: " + name);
    auto ty = wasmTypeToLLVM(curr->type);
    LOG(" .   Type: ");
    DUMP(ty);

    uint64_t size;
    if (curr->hasMax()) {
      size = curr->max;
    } else {
      size = curr->initial;
    }

    auto arrayTy = ArrayType::get(ty, size);
    auto linkage = GlobalVariable::ExternalLinkage;
    Constant* initializer = nullptr;
    if (!curr->imported()) {
      linkage = GlobalVariable::InternalLinkage;
      std::vector<Constant*> funcArray;

      for (size_t i = 0; i < size; i++) {
        funcArray.push_back(nullFunc);
      }

      initializer = ConstantArray::get(ArrayType::get(funcRef, size),
                                       ArrayRef<Constant*>(funcArray));
    }
    auto* table =
      new GlobalVariable(*mod, arrayTy, false, linkage, initializer, name);

    LOG(" .  Table: ");
    DUMP(table);
  }

  void visitGlobal(wasm::Global* curr) {
    if (curr->imported()) {
      visitImportedGlobal(curr);
    } else {
      visitDefinedGlobal(curr);
    }
  }

  GlobalVariable* emitGlobalType(wasm::Global* curr,
                                 GlobalValue::LinkageTypes linkage,
                                 Constant* init) {
    auto ty = wasmTypeToLLVM(curr->type);
    if (ty) {
      return new GlobalVariable(
        *mod, ty, !curr->mutable_, linkage, init, curr->name.toString());
    } else {
      PANIC("Type not found: " + curr->type.toString());
    }
    return nullptr;
  }

  void visitImportedGlobal(wasm::Global* curr) {
    LOG("Import the globals " + curr->name.toString());
    emitGlobalType(curr, GlobalVariable::ExternalLinkage, nullptr);
  }

  void visitDefinedGlobal(wasm::Global* curr) {
    auto name = curr->name.toString();
    LOG("Define the globals " + name);

    Constant* init;
    if (curr->init->is<wasm::Const>()) {
      init = emitConstExpression(curr->init);
      emitGlobalType(curr, GlobalVariable::InternalLinkage, init);
    } else {
      auto type = wasmTypeToLLVM(curr->init->type);
      init = Constant::getNullValue(type);

      auto global = emitGlobalType(curr, GlobalVariable::InternalLinkage, init);

      auto [initFunc, entry] = getDataCtor();

      auto prevFunc = stash_function(initFunc);
      builder->SetInsertPoint(entry);

      emitter->appendInsertionBlock("Init_" + name);

      auto initExpr = emitter->visit(curr->init);
      LOG(" .    Init for: " + name);
      DUMP(initExpr);

      builder->CreateStore(initExpr, global);

      restore_function(prevFunc);
    }
  }

  Constant* emitConstExpression(wasm::Expression* curr) {
    if (curr->is<wasm::Const>()) {
      LOG(std::string("Handle constant of Type: ") + curr->type.toString());
      auto type = wasmTypeToLLVM(curr->type);
      if (type) {
        wasm::Literal val = getLiteralFromConstExpression(curr);
        if (type->isIntegerTy()) {
          return ConstantInt::get(type, val.getInteger());
        } else if (type->isFloatingPointTy()) {
          return ConstantFP::get(type, val.getFloat());
        }
      }
    }
    PANIC("TODO: Handle none constant expression");
    return nullptr;
  }

  void visitFunction(wasm::Function* curr) {
    if (curr->imported()) {
      visitImportedFunction(curr);
    } else {
      visitDefinedFunction(curr);
    }
  }

  void visitFunctionPreview(wasm::Function* curr) {
    auto name = curr->name.toString();
    LOG("Visit the signautre of function " + curr->name.toString());

    auto linkage = GlobalValue::InternalLinkage;
    if (curr->imported()) {
      linkage = GlobalValue::ExternalLinkage;
    }

    createFunction(curr, linkage);
  }

  inline std::string idxToName(int localIdx) {
    return "_" + std::to_string(localIdx);
  }

  inline std::string argToName(int localIdx) {
    return "arg_" + std::to_string(localIdx);
  }

  Function* createFunction(FunctionType* funcType,
                           GlobalValue::LinkageTypes linkage,
                           const std::string& name) {
    auto func = Function::Create(funcType, linkage, name, *mod);

    for (size_t i = 0; i < func->arg_size(); i++) {
      auto arg = func->getArg(i);
      if (arg != nullptr) {
        arg->setName(argToName(i));
      }
    }

    namedFunctions[name] = func;

    return func;
  }

  Function* createFunction(wasm::Function* curr,
                           GlobalValue::LinkageTypes linkage) {
    auto paramTypes = wasmTupleToLLVMTypes(curr->getParams());
    LOG("Args count: " + std::to_string(paramTypes.size()));
    auto retType = wasmTypeToLLVM(curr->getResults());
    if (!retType) {
      retType = builder->getVoidTy();
    }
    auto funcType = FunctionType::get(retType, paramTypes, false);
    return createFunction(funcType, linkage, curr->name.toString());
  }

  void visitImportedFunction(wasm::Function* curr) {
    LOG("Declar the external function " + curr->name.toString());
    auto func = namedFunctions[curr->name.toString()];
    if (func == nullptr) {
      createFunction(curr, GlobalValue::ExternalLinkage);
      PANIC(" .  The function is not predefined");
    }
  }

  void finishFunction(Function& func) {
    if (func.size() == 0) {
      return;
    }

    BasicBlock* lastBB = builder->GetInsertBlock();
    if (!lastBB->getTerminator()) {
      if (lastValue != nullptr && !func.getReturnType()->isVoidTy()) {
        builder->CreateRet(lastValue);
      } else {
        builder->CreateRetVoid();
      }
    }
    LOG("Verifying Function: " + func.getName().str());
    if (verifyFunction(func, &errs())) {
      DUMP(&func);
      PANIC("Invalid Function " + func.getName().str());
    }
  }

  void visitDefinedFunction(wasm::Function* curr) {
    LOG("Define the function " + curr->name.toString());

    auto func = namedFunctions[curr->name.toString()];
    if (func == nullptr) {
      func = createFunction(curr, GlobalValue::InternalLinkage);
      PANIC(" .  The function is not predefined");
    }
    auto prevFunc = stash_function(func);

    static std::string entryName("entry_Alloca");
    auto entry = BasicBlock::Create(*context, entryName, func);
    builder->SetInsertPoint(entry);
    namedBlocks[entryName] = entry;

    for (size_t i = 0; i < curr->getVarIndexBase(); i++) {
      auto arg = func->getArg(i);
      Type* type = arg->getType();
      if (!type->isVoidTy()) {
        auto varName = idxToName(i);
        LOG(" .   Populate Local Var for Arg: " + varName + " - type: ");
        DUMP(type);
        auto ptr = builder->CreateAlloca(type, nullptr, varName);
        builder->CreateStore(arg, ptr);
      }
    }

    for (size_t i = curr->getVarIndexBase(); i < curr->getNumLocals(); i++) {
      Type* type = wasmTypeToLLVM(curr->getLocalType(i));
      if (!type->isVoidTy()) {
        auto varName = idxToName(i);
        LOG(" .   Create Local Var: " + varName + " - type: ");
        DUMP(type);
        builder->CreateAlloca(type, nullptr, varName);
      }
    }

    // ValueSymbolTable* symtab = func->getValueSymbolTable();
    // if (symtab != nullptr) {
    //   auto* val = symtab->lookup(idxToName(curr->getVarIndexBase()));
    //   if (val != nullptr) {
    //     LOG(" . Found LocalSymbol: " + val->getName().str());
    //     DUMP(val);
    //   }

    //   for (auto ai = symtab->begin(), ae = symtab->end(); ai != ae; ai++) {
    //     auto name = ai->first();
    //     auto value = ai->second;
    //     LOG("  > symbol: " + name.str());
    //     DUMP(value);
    //   }
    // }

    if (!curr->stackIR) {

      emitter->iterateBody(curr->body);

      finishFunction(*func);

    } else {
      PANIC("StackIR is not supported");
    }

    restore_function(prevFunc);
  }

  void visit(wasm::Expression* curr) {
    LOG(std::string("Visit Expression of Type: ") + curr->type.toString());
  }

  void visitTag(wasm::Tag* curr) {
    LOG("Figure out what the tag means  " + curr->name.toString());
    if (curr->imported()) {
      visitImportedTag(curr);
    } else {
      visitDefinedTag(curr);
    }
  }

  void visitImportedTag(wasm::Tag* curr) {
    LOG("Figure out what the tag means");
  }

  void visitDefinedTag(wasm::Tag* curr) {
    LOG("Figure out what the tag means");
  }

  const std::string initDataCtor = "__wasm2llvm_init_data";

  typedef std::tuple<Function*, BasicBlock*> FuncStash;

  FuncStash getDataCtor() {
    auto initFunc = namedFunctions[initDataCtor];
    BasicBlock* entry;

    if (initFunc == nullptr) {
      initFunc = createFunction(
        FunctionType::get(v01d, false), Function::PrivateLinkage, initDataCtor);
      entry = BasicBlock::Create(*context, "entry", initFunc);
    } else {
      entry = &initFunc->back();
    }

    return std::make_tuple(initFunc, entry);
  }

  void visitDataSegment(wasm::DataSegment* curr) {
    LOG("visitDataSegment to define the data");

    auto name = dataName(curr->name.toString());
    LOG(" .   Name: " + name);

    auto memName = memoryName(curr->memory.toString());
    LOG(" .   Memory: " + memName);

    auto& memory = namedMemories[memName];

    if (!curr->isPassive) {
      size_t size = curr->data.size();

      auto mem_ty = IntegerType::get(*context, 8);
      auto arrayTy = ArrayType::get(mem_ty, size);
      auto linkage = GlobalVariable::PrivateLinkage;

      std::vector<Constant*> data(size);

      std::transform(curr->data.begin(),
                     curr->data.end(),
                     data.begin(),
                     [&](char c) { return ConstantInt::get(mem_ty, c); });

      auto initializer = ConstantArray::get(arrayTy, ArrayRef<Constant*>(data));
      auto globalData =
        new GlobalVariable(*mod, arrayTy, false, linkage, initializer, name);

      auto [initFunc, entry] = getDataCtor();

      auto prevFunc = stash_function(initFunc);
      builder->SetInsertPoint(entry);
      emitter->appendInsertionBlock("Init_" + name);

      auto offset = emitter->visit(curr->offset);
      LOG(" .    Offset: ");
      DUMP(offset);

      auto dest =
        builder->CreateGEP(IntegerType::get(*context, 8), memory.ptr, offset);
      builder->CreateMemCpy(dest, Align(1), globalData, Align(1), size);

      restore_function(prevFunc);
    } else {
      LOG(" .   Passive");
    }
  }

  FuncStash stash_function(Function* func) {
    auto prevFunc = currentFunc;
    currentFunc = func;
    lastValue = nullptr;
    namedBlocks.clear();
    branchMap.clear();
    return {prevFunc, builder->GetInsertBlock()};
  }

  void restore_function(FuncStash& stash) {
    auto [func, block] = stash;
    currentFunc = func;
    lastValue = nullptr;
    namedBlocks.clear();
    branchMap.clear();
    builder->SetInsertPoint(block);
  }

  void visitElementSegment(wasm::ElementSegment* curr) {
    LOG("visitElementSegment to define the element segment");
    auto name = curr->name.toString();
    LOG(" .  Segement name: " + name);

    auto tname = tableName(curr->table.toString());
    LOG(" .   Table Name: " + tname);
    auto table = mod->getNamedGlobal(tname);
    DUMP(table);

    if (curr->offset->is<wasm::Const>()) {
      auto offset = dyn_cast<ConstantInt>(emitter->visit(curr->offset));
      if (offset == nullptr) {
        PANIC(" .     Offset is not const int");
      }
      LOG(" .    Offset: ");
      DUMP(offset);

      auto idx = *offset->getValue().getRawData();
      LOG(" .  Data: ");
      std::vector<Constant*> funcArray;

      if (idx > 0) {
        for (size_t i = 0; i < idx; i++) {
          funcArray.push_back(nullFunc);
        }
      }

      for (auto* entry : curr->data) {
        auto data = emitter->visit(entry);
        DUMP(data);
        funcArray.push_back(dyn_cast<Function>(data));
      }

      table->setInitializer(
        ConstantArray::get(ArrayType::get(funcRef, funcArray.size()),
                           ArrayRef<Constant*>(funcArray)));
    } else {
      auto fname = "__wasm2llvm_init_seg_" + name;
      auto func = createFunction(
        FunctionType::get(v01d, false), Function::PrivateLinkage, fname);

      auto prevFunc = stash_function(func);

      auto entry = BasicBlock::Create(*context, "entry", func);
      builder->SetInsertPoint(entry);

      auto offset = emitter->visit(curr->offset);

      for (auto* entry : curr->data) {
        auto data = emitter->visit(entry);
        DUMP(data);
        auto ptr = builder->CreateGEP(data->getType(), table, offset);
        builder->CreateStore(data, ptr);
        offset = builder->CreateAdd(offset, ConstantInt::get(i32, 1));
      }

      finishFunction(*func);

      restore_function(prevFunc);

      llvm::appendToGlobalCtors(*mod, func, 0);
    }
  }

  void visitExport(wasm::Export* curr) {
    LOG("Mark exports external " + curr->name.toString() +
        " == " + curr->value.toString());
    auto localName = curr->value.toString();
    GlobalVariable* global = nullptr;

    switch (curr->kind) {
      case wasm::ExternalKind::Function: {
        auto func = namedFunctions[localName];
        if (func) {
          func->setLinkage(Function::ExternalLinkage);
        }
        break;
      }
      case wasm::ExternalKind::Table: {
        global = mod->getNamedGlobal(tableName(localName));
        break;
      }
      case wasm::ExternalKind::Memory: {
        global = mod->getNamedGlobal(memoryName(localName));
        break;
      }
      default: {
        global = mod->getNamedGlobal(localName);
      }
    }

    if (global) {
      global->setLinkage(GlobalValue::ExternalLinkage);
    }
  }

  void finishMemory() {
    LOG("FinishMemory()");
    auto initFunc = namedFunctions[initDataCtor];
    if (initFunc != nullptr) {
      auto lastBB = &initFunc->back();
      builder->SetInsertPoint(lastBB);
      finishFunction(*initFunc);
      llvm::appendToGlobalCtors(*mod, initFunc, 0);
    }
  }

  void visitModule(wasm::Module* curr) {
    LOG("Create Module: " + curr->name.toString());

    mod = std::make_unique<Module>(
      curr->name.is() ? curr->name.toString() : "WasmModule", *context);
    mod->setTargetTriple(triple.getTriple());
    builder = std::make_unique<IRBuilder<>>(*context);

    auto nullFuncType = FunctionType::get(i32, false);
    static const char* null_func_name = "__wasm2llvm_table_null_func";
    nullFunc = createFunction(
      nullFuncType, GlobalValue::InternalLinkage, null_func_name);
    auto entry = BasicBlock::Create(*context, "", nullFunc);
    builder->SetInsertPoint(entry);
    builder->CreateRet(Constant::getNullValue(i32));

    auto indexedTypes = wasm::ModuleUtils::getOptimizedIndexedHeapTypes(*curr);
    for (auto type : indexedTypes.types) {
      handleHeapType(type);
    }

    // Scan for all functions to define the function signatures
    wasm::ModuleUtils::iterImportedFunctions(
      *curr, [&](wasm::Function* func) { visitFunctionPreview(func); });
    wasm::ModuleUtils::iterDefinedFunctions(
      *curr, [&](wasm::Function* func) { visitFunctionPreview(func); });

    wasm::ModuleUtils::iterImportedMemories(
      *curr, [&](wasm::Memory* memory) { visitMemory(memory); });
    wasm::ModuleUtils::iterImportedTables(
      *curr, [&](wasm::Table* table) { visitTable(table); });
    wasm::ModuleUtils::iterImportedGlobals(
      *curr, [&](wasm::Global* global) { visitGlobal(global); });
    wasm::ModuleUtils::iterImportedFunctions(
      *curr, [&](wasm::Function* func) { visitFunction(func); });
    wasm::ModuleUtils::iterImportedTags(*curr,
                                        [&](wasm::Tag* tag) { visitTag(tag); });
    wasm::ModuleUtils::iterDefinedGlobals(
      *curr, [&](wasm::Global* global) { visitGlobal(global); });
    wasm::ModuleUtils::iterDefinedMemories(
      *curr, [&](wasm::Memory* memory) { visitMemory(memory); });
    for (auto& segment : curr->dataSegments) {
      visitDataSegment(segment.get());
    }
    wasm::ModuleUtils::iterDefinedTables(
      *curr, [&](wasm::Table* table) { visitTable(table); });
    for (auto& segment : curr->elementSegments) {
      visitElementSegment(segment.get());
    }

    auto elemDeclareNames =
      wasm::TableUtils::getFunctionsNeedingElemDeclare(*curr);
    if (!elemDeclareNames.empty()) {
      // LOG: Do we need to delcare the function before using it?
    }
    wasm::ModuleUtils::iterDefinedTags(*curr,
                                       [&](wasm::Tag* tag) { visitTag(tag); });
    for (auto& child : curr->exports) {
      visitExport(child.get());
    }
    if (curr->start.is()) {
      // LOG: export the start function
    }
    wasm::ModuleUtils::iterDefinedFunctions(
      *curr, [&](wasm::Function* func) { visitFunction(func); });

    finishMemory();

    LOG("Verifying Module: " + mod->getName().str());
    if (verifyModule(*mod, &errs())) {
      DUMP(mod);
      PANIC("Invalid Module: " + mod->getName().str());
    }
  }

  // Pass entry point. LOG: parallelize?

  void run(wasm::Module* module) override {
    initLLVM();
    initPassInstance();

    visitModule(module);

    std::string llvm_output = "stdout";
    if (getPassOptions().arguments.count("llvm-output")) {
      llvm_output = getPassOptions().arguments["llvm-output"];
    }

    if (llvm_output == "stdout") {
      llvm::raw_os_ostream out_steam(std::cout);
      mod->print(out_steam, nullptr);
    } else {
      std::error_code EC;
      llvm::raw_fd_ostream out_steam(llvm_output, EC);
      if (EC) {
        wasm::Fatal() << "Failed to open output file: " << llvm_output;
      }
      mod->print(out_steam, nullptr);
    }
  }

  const std::vector<Type*> wasmTupleToLLVMTypes(wasm::Type type) {
    std::vector<Type*> llvmTypes{};
    auto types =
      (type.isTuple() ? type.getTuple().types : std::vector<wasm::Type>{type});

    for (auto t : types) {
      Type* llvmType = wasmTypeToLLVM(t);
      if (!llvmType->isVoidTy()) {
        llvmTypes.push_back(llvmType);
      }
    }

    return llvmTypes;
  }

  // Returns the LLVM type for a wasm type, if there is one.
  Type* wasmTypeToLLVM(wasm::Type type) {
    if (type == wasm::Type::i32) {
      return i32;
    }
    if (type == wasm::Type::i64) {
      return i64;
    }
    if (type == wasm::Type::f32) {
      return f32;
    }
    if (type == wasm::Type::f64) {
      return f64;
    }

    if (type == wasm::Type::none || type == wasm::Type::unreachable) {
      return v01d;
    }

    if (type.isRef()) {
      if (type.isFunction()) {
        return funcRef;
      }
    }

    PANIC("TODO: Support Type " + type.toString());
    return v01d;
  }

  FunctionType* wasmSignatureToLLVM(wasm::Signature sig) {
    LOG("wasmSignatureToLLVM(): " + sig.toString());
    auto paramTypes = wasmTupleToLLVMTypes(sig.params);
    LOG(" .  Args count: " + std::to_string(paramTypes.size()));
    auto retType = wasmTypeToLLVM(sig.results);
    if (!retType) {
      retType = builder->getVoidTy();
    }
    auto funcType = FunctionType::get(retType, paramTypes, false);
    LOG(" .  Function Type:");
    DUMP(funcType);

    return funcType;
  }
};

namespace wasm {

Pass* createLLVMPass() { return new LLVMPass(); }

void printLLVM(wasm::Module& module, std::ostream& output) {
  LLVMPass pass(output);

  pass.run(&module);
}

} // namespace wasm

#define Parent (this->parent)
#define Builder (*Parent.builder)
#define Context (*Parent.context)
#define Module (*Parent.mod)
#define Func (Parent.currentFunc)

BasicBlock* LLVMPass::IREmitter::createBlock(const std::string& name,
                                             Function* parent) {
  BasicBlock* block = BasicBlock::Create(*Parent.context, name, parent);
  Parent.namedBlocks[name] = block;
  return block;
}

void LLVMPass::IREmitter::appendInsertionBlock(BasicBlock* block) {
  BasicBlock* prevInstPt = Builder.GetInsertBlock();

  if (prevInstPt->getTerminator() == nullptr) {
    Builder.CreateBr(block);
    Parent.branchMap[block].push_back({prevInstPt, Parent.lastValue});
  }
  Func->getBasicBlockList().push_back(block);
  Builder.SetInsertPoint(block);
}

void LLVMPass::IREmitter::appendInsertionBlock(const std::string& name) {
  auto block = createBlock(name);
  appendInsertionBlock(block);
}

Value* LLVMPass::IREmitter::visitBlock(wasm::Block* curr) {
  LOG("  ----> IREmitter::visitBlock: ");

  auto name = curr->name.toString();
  BasicBlock* prevInstPt = Builder.GetInsertBlock();
  BasicBlock* block = createBlock(name + "|start", Func);

  if (prevInstPt->getTerminator() == nullptr) {
    Builder.CreateBr(block);
  }

  BasicBlock* blockEnd = createBlock(name);

  Builder.SetInsertPoint(block);

  LOG(
    "  ----> IREmitter::visitBlock: " +
    (curr->name.isNull() ? "(null name)" : name) +
    (curr->type.isConcrete() ? " (result " + curr->type.toString() + ")" : ""));

  for (auto item : curr->list) {
    this->visit(item);
  }

  appendInsertionBlock(blockEnd);

  LOG("  <----- IREmitter::doneWithBlock: " + name);

  if (curr->type.isConcrete()) {
    LOG("   Block has result type: " + curr->type.toString());
    auto type = Parent.wasmTypeToLLVM(curr->type);

    auto branches = Parent.branchMap[blockEnd];
    if (branches.size() > 1) {
      LOG("   Block branch cnt: " + std::to_string(branches.size()));
      auto phi = Builder.CreatePHI(type, branches.size());
      for (auto [from, value] : branches) {
        LOG(" .   adding Branch from : " + from->getName().str());
        DUMP(value);

        phi->addIncoming(value, from);
      }
      return phi;
    } else {
      auto value = Parent.lastValue;
      if (value->getType() != type) {
        value = Constant::getNullValue(type);
      }
      return value;
    }
  }

  return blockEnd;
}

Value* LLVMPass::IREmitter::assurePtrType(Value* val, Type* elmTy) {
  Value* ptr = nullptr;
  auto ty = val->getType();
  auto ptrTy =
    elmTy ? PointerType::getUnqual(elmTy) : PointerType::getUnqual(Context);

  if (ty->isIntegerTy()) {
    ptr = new IntToPtrInst(val, ptrTy, "", Builder.GetInsertBlock());
  } else if (ty->isPointerTy()) {
    if (ty == ptrTy) {
      ptr = val;
    } else {
      ptr = new BitCastInst(val, ptrTy, "", Builder.GetInsertBlock());
    }
  } else {
    DUMP(val);
    PANIC(" -- Invalid Casting to Pointer ");
  }

  return ptr;
}

Value* LLVMPass::IREmitter::assureIntType(Value* val, int bits) {
  auto ty = val->getType();
  if (ty->isIntegerTy() || ty->isPointerTy()) {
    if (ty->getPrimitiveSizeInBits() != (uint64_t)bits) {
      val = CastInst::CreateIntegerCast(val,
                                        IntegerType::get(Context, bits),
                                        false,
                                        "",
                                        Builder.GetInsertBlock());
    }
  } else {
    DUMP(ty);
    DUMP(val);
    PANIC(" -- Invalid Casting to Integer of " + std::to_string(bits) +
          " bits");
  }
  return val;
}

Value* LLVMPass::IREmitter::assureType(Value* val, Type* type) {
  auto ty = val->getType();

  if (ty != type) {
    DUMP(val);
    PANIC(" -- Invalid type ");
  }

  return val;
}

Value* LLVMPass::IREmitter::visitMemoryFill(wasm::MemoryFill* curr) {
  LOG("  ----> IREmitter::visitMemoryFill: ");

  auto name = Parent.memoryName(curr->memory.toString());
  auto& memory = Parent.namedMemories[name];
  LOG(" .     Memory: " + name);

  auto offset = visit(curr->dest);

  auto ptr =
    Builder.CreateGEP(IntegerType::get(Context, 8), memory.ptr, offset);

  auto val = CastInst::CreateIntegerCast(visit(curr->value),
                                         IntegerType::get(Context, 8),
                                         false,
                                         "",
                                         Builder.GetInsertBlock());
  auto size = visit(curr->size);
  LOG("       Ptr:");
  DUMP(ptr);
  LOG("       Val:");
  DUMP(val);
  LOG("       Size:");
  DUMP(size);
  auto inst = Builder.CreateMemSet(ptr, val, size, Align(1));
  return inst;
}

Value* LLVMPass::IREmitter::visitConst(wasm::Const* curr) {
  LOG("  ----> IREmitter::visitConst: ");
  auto ty = Parent.wasmTypeToLLVM(curr->type);
  if (ty) {
    if (ty->isIntegerTy()) {
      return ConstantInt::get(ty, curr->value.getInteger());
    } else if (ty->isFloatingPointTy()) {
      return ConstantFP::get(ty, curr->value.getFloat());
    }
  }
  PANIC("Type not found: " + curr->type.toString());
  return nullptr;
}

static Value* getNamedValue(Function* func, std::string name) {
  Value* val = nullptr;
  ValueSymbolTable* symtab = func->getValueSymbolTable();
  if (symtab != nullptr) {
    val = symtab->lookup(name);
    if (val != nullptr) {
      LOG(" . Found LocalSymbol: " + val->getName().str());
      DUMP(val);
    }
  }

  return val;
}

Value* LLVMPass::IREmitter::visitLocalGet(wasm::LocalGet* curr) {
  LOG("  ----> IREmitter::visitLocalGet: ");
  auto name = Parent.idxToName(curr->index);
  auto var = getNamedValue(Func, name);
  if (!var) {
    PANIC("Local Var not found: " + std::to_string(curr->index));
  } else {
    if (var->getType()->isPointerTy()) {
      auto ty = Parent.wasmTypeToLLVM(curr->type);
      // auto ty = var->getType()->getPointerElementType();
      var = Builder.CreateLoad(ty, var);
    }
  }
  return var;
}

Value* LLVMPass::IREmitter::visitLocalSet(wasm::LocalSet* curr) {
  LOG("  ----> IREmitter::visitLocalSet: ");
  auto name = Parent.idxToName(curr->index);
  auto var = getNamedValue(Func, name);
  if (!var) {
    PANIC("Local Var not found: " + std::to_string(curr->index));
  } else {
    LOG(" .     Local Var found: " + name);
    DUMP(var);
    if (var->getType()->isPointerTy()) {
      auto val = visit(curr->value);
      auto inst = Builder.CreateStore(val, var);
      if (curr->isTee()) {
        auto ty = Parent.wasmTypeToLLVM(curr->type);
        var = Builder.CreateLoad(ty, var);
      } else {
        var = inst;
      }
    }
  }
  return var;
}

Value* LLVMPass::IREmitter::visitStore(wasm::Store* curr) {
  LOG("  ----> IREmitter::visitStore: ");
  int bits = curr->valueType.getByteSize() * 8;

  if (curr->bytes < 4 ||
      (curr->valueType == wasm::Type::i64 && curr->bytes < 8)) {
    switch (curr->bytes) {
      case 1: {
        bits = 8;
        break;
      }
      case 2: {
        bits = 16;
        break;
      }
      case 4: {
        bits = 32;
        break;
      }
      default: {
        PANIC("Invalid Store strip down bytes = " +
              std::to_string(curr->bytes));
      }
    }
  }

  auto instTy = Parent.wasmTypeToLLVM(curr->valueType);
  Type* ty;
  if (instTy->isIntegerTy()) {
    ty = IntegerType::get(Context, bits);
  } else if (instTy->isFloatingPointTy()) {
    ty = instTy;
  } else {
    LOG(" .  Storing Type: " + curr->valueType.toString());
    DUMP(instTy);
    PANIC(" .   Not supported");
  }

  auto ptr = assurePtrType(visit(curr->ptr), ty);

  if (curr->offset != 0) {
    auto offset = ConstantInt::get(Parent.i32, curr->offset / (bits / 8));
    ptr = Builder.CreateGEP(ty, ptr, offset);
  }

  auto val = visit(curr->value);

  if (curr->bytes != curr->valueType.getByteSize()) {
    val = assureIntType(val, bits);
  }

  LOG("       Ptr:");
  DUMP(ptr);
  LOG("       Val:");
  DUMP(val);
  Builder.CreateStore(val, ptr);
  return nullptr;
}

Value* LLVMPass::IREmitter::visitLoad(wasm::Load* curr) {
  LOG("  ----> IREmitter::visitLoad: ");
  int bits = curr->type.getByteSize() * 8;
  if (curr->bytes < 4 || (curr->type == wasm::Type::i64 && curr->bytes < 8)) {
    switch (curr->bytes) {
      case 1: {
        bits = 8;
        break;
      }
      case 2: {
        bits = 16;
        break;
      }
      case 4: {
        bits = 32;
        break;
      }
      default: {
        PANIC("Invalid Store strip down bytes = " +
              std::to_string(curr->bytes));
      }
    }
  }

  auto instType = Parent.wasmTypeToLLVM(curr->type);
  Type* ty;
  if (instType->isIntegerTy()) {
    ty = IntegerType::get(Context, bits);
  } else if (instType->isFloatingPointTy()) {
    ty = instType;
  } else {
    LOG(" .  Loading Type: " + curr->type.toString());
    DUMP(instType);
    PANIC(" .   Not supported");
  }
  auto ptr = assurePtrType(visit(curr->ptr), ty);

  if (curr->offset != 0) {
    auto offset = ConstantInt::get(Parent.i32, curr->offset / (bits / 8));
    ptr = Builder.CreateGEP(ty, ptr, offset);
  }

  Value* val = Builder.CreateLoad(ty, ptr);
  if (curr->bytes != curr->type.getByteSize()) {
    val = assureIntType(val, curr->type.getByteSize() * 8);
  }

  return val;
}

Function*
LLVMPass::IREmitter::assureFunction(const std::string& fname,
                                    const std::vector<Type*>& paramTypes) {
  LOG(" . IREmitter::assureFunction " + fname);
  LOG(" .   Args count: " + std::to_string(paramTypes.size()));

  // Set the default return type to i32. May need to change when the actual
  // function is defined.
  auto funcType = FunctionType::get(Parent.i32, paramTypes, false);
  auto func =
    Parent.createFunction(funcType, GlobalVariable::InternalLinkage, fname);

  for (size_t i = 0; i < func->arg_size(); i++) {
    auto arg = func->getArg(i);
    if (arg != nullptr) {
      arg->setName(Parent.idxToName(i));
    }
  }

  return func;
}

Value* LLVMPass::IREmitter::rotl(Value* a, Value* b) {
  auto ty = a->getType();
  if (!ty->isIntegerTy()) {
    PANIC("Rotl on non-Integer");
  }
  auto bits = ConstantInt::get(ty, ty->getIntegerBitWidth());
  auto r = Builder.CreateURem(b, bits);
  auto tmp0 = Builder.CreateShl(a, r);
  auto tmp1 = Builder.CreateSub(bits, r);
  auto tmp2 = Builder.CreateLShr(a, tmp1);
  auto tmp3 = Builder.CreateOr(tmp0, tmp2);

  return tmp3;
}

Value* LLVMPass::IREmitter::rotr(Value* a, Value* b) {
  auto ty = a->getType();
  if (!ty->isIntegerTy()) {
    PANIC("Rotr on non-Integer");
  }
  auto bits = ConstantInt::get(ty, ty->getIntegerBitWidth());
  auto r = Builder.CreateURem(b, bits);
  auto tmp0 = Builder.CreateLShr(a, r);
  auto tmp1 = Builder.CreateSub(bits, r);
  auto tmp2 = Builder.CreateShl(a, tmp1);
  auto tmp3 = Builder.CreateOr(tmp0, tmp2);

  return tmp3;
}

Value* LLVMPass::IREmitter::eqz(Value* v) {
  auto ty = v->getType();
  if (!ty->isIntegerTy()) {
    PANIC("Eqz on non-Integer");
  }

  auto inst = Builder.CreateICmpNE(ConstantInt::get(ty, 0), v);

  return inst;
}

Value* LLVMPass::IREmitter::visitCall(wasm::Call* curr) {
  LOG("  ----> IREmitter::visitCall: ");
  auto fname = curr->target.toString();
  auto func = Module.getFunction(fname);

  if (!func) {
    auto fname = curr->target.toString();
    std::vector<Type*> paramTypes;
    for (auto it = curr->operands.begin(); it != curr->operands.end(); it++) {
      wasm::Expression* op = *it;
      paramTypes.push_back(Parent.wasmTypeToLLVM(op->type));
    }
    func = assureFunction(fname, paramTypes);
  }

  std::vector<Value*> args;
  for (auto it = curr->operands.begin(); it != curr->operands.end(); it++) {
    wasm::Expression* op = *it;
    args.push_back(visit(op));
  }

  auto inst = Builder.CreateCall(func, args);
  inst->setTailCall(curr->isReturn);

  return inst;
}

Value* LLVMPass::IREmitter::visitBinary(wasm::Binary* curr) {
  LOG("  ----> IREmitter::visitBinary: ");
  auto left = visit(curr->left);
  auto right = visit(curr->right);

  LOG("   Left Operand: ");
  DUMP(left);
  LOG("   Right Operand: ");
  DUMP(right);

  Value* inst = nullptr;

  switch (curr->op) {
    case wasm::AddInt32:
      LOG("   op = i32.add");
      inst = Builder.CreateAdd(left, right);
      break;
    case wasm::SubInt32:
      LOG("   op = i32.sub");
      inst = Builder.CreateSub(left, right);
      break;
    case wasm::MulInt32:
      LOG("   op = i32.mul");
      inst = Builder.CreateMul(left, right);
      break;
    case wasm::DivSInt32:
      LOG("   op = i32.div_s");
      inst = Builder.CreateSDiv(left, right);
      break;
    case wasm::DivUInt32:
      LOG("   op = i32.div_u");
      inst = Builder.CreateUDiv(left, right);
      break;
    case wasm::RemSInt32:
      LOG("   op = i32.rem_s");
      inst = Builder.CreateSRem(left, right);
      break;
    case wasm::RemUInt32:
      LOG("   op = i32.rem_u");
      inst = Builder.CreateURem(left, right);
      break;
    case wasm::AndInt32:
      LOG("   op = i32.and");
      inst = Builder.CreateAnd(left, right);
      break;
    case wasm::OrInt32:
      LOG("   op = i32.or");
      inst = Builder.CreateOr(left, right);
      break;
    case wasm::XorInt32:
      LOG("   op = i32.xor");
      inst = Builder.CreateXor(left, right);
      break;
    case wasm::ShlInt32:
      LOG("   op = i32.shl");
      inst = Builder.CreateShl(left, right);
      break;
    case wasm::ShrUInt32:
      LOG("   op = i32.shr_u");
      inst = Builder.CreateLShr(left, right);
      break;
    case wasm::ShrSInt32:
      LOG("   op = i32.shr_s");
      inst = Builder.CreateAShr(left, right);
      break;
    case wasm::RotLInt32:
      LOG("   op = i32.rotl");
      // inst = rotl(left, right);
      inst = Builder.CreateIntrinsic(
        Intrinsic::fshl, Parent.i32, ArrayRef<Value*>{left, left, right});
      break;
    case wasm::RotRInt32:
      LOG("   op = i32.rotr");
      // inst = rotr(left, right);
      inst = Builder.CreateIntrinsic(
        Intrinsic::fshr, Parent.i32, ArrayRef<Value*>{left, left, right});
      break;
    case wasm::EqInt32:
      LOG("   op = i32.eq");
      inst = Builder.CreateICmpEQ(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::NeInt32:
      LOG("   op = i32.ne");
      inst = Builder.CreateICmpNE(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::LtSInt32:
      LOG("   op = i32.lt_s");
      inst = Builder.CreateICmpSLT(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::LtUInt32:
      LOG("   op = i32.lt_u");
      inst = Builder.CreateICmpULT(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::LeSInt32:
      LOG("   op = i32.le_s");
      inst = Builder.CreateICmpSLE(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::LeUInt32:
      LOG("   op = i32.le_u");
      inst = Builder.CreateICmpULE(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::GtSInt32:
      LOG("   op = i32.gt_s");
      inst = Builder.CreateICmpUGT(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::GtUInt32:
      LOG("   op = i32.gt_u");
      inst = Builder.CreateICmpUGT(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::GeSInt32:
      LOG("   op = i32.ge_s");
      inst = Builder.CreateICmpSGE(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::GeUInt32:
      LOG("   op = i32.ge_u");
      inst = Builder.CreateICmpUGE(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;

    case wasm::AddInt64:
      LOG("   op = i64.add");
      inst = Builder.CreateAdd(left, right);
      break;
    case wasm::SubInt64:
      LOG("   op = i64.sub");
      inst = Builder.CreateSub(left, right);
      break;
    case wasm::MulInt64:
      LOG("   op = i64.mul");
      inst = Builder.CreateMul(left, right);
      break;
    case wasm::DivSInt64:
      LOG("   op = i64.div_s");
      inst = Builder.CreateSDiv(left, right);
      break;
    case wasm::DivUInt64:
      LOG("   op = i64.div_u");
      inst = Builder.CreateUDiv(left, right);
      break;
    case wasm::RemSInt64:
      LOG("   op = i64.rem_s");
      inst = Builder.CreateSRem(left, right);
      break;
    case wasm::RemUInt64:
      LOG("   op = i64.rem_u");
      inst = Builder.CreateURem(left, right);
      break;
    case wasm::AndInt64:
      LOG("   op = i64.and");
      inst = Builder.CreateAnd(left, right);
      break;
    case wasm::OrInt64:
      LOG("   op = i64.or");
      inst = Builder.CreateOr(left, right);
      break;
    case wasm::XorInt64:
      LOG("   op = i64.xor");
      inst = Builder.CreateXor(left, right);
      break;
    case wasm::ShlInt64:
      LOG("   op = i64.shl");
      inst = Builder.CreateShl(left, right);
      break;
    case wasm::ShrUInt64:
      LOG("   op = i64.shr_u");
      inst = Builder.CreateLShr(left, right);
      break;
    case wasm::ShrSInt64:
      LOG("   op = i64.shr_s");
      inst = Builder.CreateAShr(left, right);
      break;
    case wasm::RotLInt64:
      LOG("   op = i64.rotl");
      // inst = rotl(left, right);
      inst = Builder.CreateIntrinsic(
        Intrinsic::fshl, Parent.i64, ArrayRef<Value*>{left, left, right});
      break;
    case wasm::RotRInt64:
      LOG("   op = i64.rotr");
      // inst = rotr(left, right);
      inst = Builder.CreateIntrinsic(
        Intrinsic::fshr, Parent.i64, ArrayRef<Value*>{left, left, right});
      break;
    case wasm::EqInt64:
      LOG("   op = i64.eq");
      inst = Builder.CreateICmpEQ(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::NeInt64:
      LOG("   op = i64.ne");
      inst = Builder.CreateICmpNE(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::LtSInt64:
      LOG("   op = i64.lt_s");
      inst = Builder.CreateICmpSLT(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::LtUInt64:
      LOG("   op = i64.lt_u");
      inst = Builder.CreateICmpULT(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::LeSInt64:
      LOG("   op = i64.le_s");
      inst = Builder.CreateICmpSLE(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::LeUInt64:
      LOG("   op = i64.le_u");
      inst = Builder.CreateICmpULE(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::GtSInt64:
      LOG("   op = i64.gt_s");
      inst = Builder.CreateICmpSGT(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::GtUInt64:
      LOG("   op = i64.gt_u");
      inst = Builder.CreateICmpUGT(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::GeSInt64:
      LOG("   op = i64.ge_s");
      inst = Builder.CreateICmpSGE(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::GeUInt64:
      LOG("   op = i64.ge_u");
      inst = Builder.CreateICmpUGE(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;

    case wasm::AddFloat32:
      LOG("   op = f32.add");
      inst = Builder.CreateFAdd(left, right);
      break;
    case wasm::SubFloat32:
      LOG("   op = f32.sub");
      inst = Builder.CreateFSub(left, right);
      break;
    case wasm::MulFloat32:
      LOG("   op = f32.mul");
      inst = Builder.CreateFMul(left, right);
      break;
    case wasm::DivFloat32:
      LOG("   op = f32.div");
      inst = Builder.CreateFDiv(left, right);
      break;
    case wasm::CopySignFloat32:
      LOG("   op = f32.copysign");
      inst = Builder.CreateBinaryIntrinsic(Intrinsic::copysign, left, right);
      break;
    case wasm::MinFloat32:
      LOG("   op = f32.min");
      inst = Builder.CreateMinimum(left, right);
      break;
    case wasm::MaxFloat32:
      LOG("   op = f32.max");
      inst = Builder.CreateMaximum(left, right);
      break;
    case wasm::EqFloat32:
      LOG("   op = f32.eq");
      inst = Builder.CreateFCmpUEQ(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::NeFloat32:
      LOG("   op = f32.ne");
      inst = Builder.CreateFCmpUNE(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::LtFloat32:
      LOG("   op = f32.lt");
      inst = Builder.CreateFCmpULT(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::LeFloat32:
      LOG("   op = f32.le");
      inst = Builder.CreateFCmpULE(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::GtFloat32:
      LOG("   op = f32.gt");
      inst = Builder.CreateFCmpUGT(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::GeFloat32:
      LOG("   op = f32.ge");
      inst = Builder.CreateFCmpUGE(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;

    case wasm::AddFloat64:
      LOG("   op = f64.add");
      inst = Builder.CreateFAdd(left, right);
      break;
    case wasm::SubFloat64:
      LOG("   op = f64.sub");
      inst = Builder.CreateFSub(left, right);
      break;
    case wasm::MulFloat64:
      LOG("   op = f64.mul");
      inst = Builder.CreateFMul(left, right);
      break;
    case wasm::DivFloat64:
      LOG("   op = f64.div");
      inst = Builder.CreateFDiv(left, right);
      break;
    case wasm::CopySignFloat64:
      LOG("   op = f64.copysign");
      inst = Builder.CreateBinaryIntrinsic(Intrinsic::copysign, left, right);
      break;
    case wasm::MinFloat64:
      LOG("   op = f64.min");
      inst = Builder.CreateMinimum(left, right);
      break;
    case wasm::MaxFloat64:
      LOG("   op = f64.max");
      inst = Builder.CreateMaximum(left, right);
      break;
    case wasm::EqFloat64:
      LOG("   op = f64.eq");
      inst = Builder.CreateFCmpUEQ(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::NeFloat64:
      LOG("   op = f64.ne");
      inst = Builder.CreateFCmpUNE(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::LtFloat64:
      LOG("   op = f64.lt");
      inst = Builder.CreateFCmpULT(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::LeFloat64:
      LOG("   op = f64.le");
      inst = Builder.CreateFCmpULE(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::GtFloat64:
      LOG("   op = f64.gt");
      inst = Builder.CreateFCmpUGT(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::GeFloat64:
      LOG("   op = f64.ge");
      inst = Builder.CreateFCmpUGE(left, right);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;

    case wasm::EqVecI8x16:
      LOG("   op = i8x16.eq");
      inst = Builder.CreateICmpEQ(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::NeVecI8x16:
      LOG("   op = i8x16.ne");
      inst = Builder.CreateICmpNE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::LtSVecI8x16:
      LOG("   op = i8x16.lt_s");
      inst = Builder.CreateICmpSLT(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::LtUVecI8x16:
      LOG("   op = i8x16.lt_u");
      inst = Builder.CreateICmpULT(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::GtSVecI8x16:
      LOG("   op = i8x16.gt_s");
      inst = Builder.CreateICmpSGT(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::GtUVecI8x16:
      LOG("   op = i8x16.gt_u");
      inst = Builder.CreateICmpUGT(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::LeSVecI8x16:
      LOG("   op = i8x16.le_s");
      inst = Builder.CreateICmpSLE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::LeUVecI8x16:
      LOG("   op = i8x16.le_u");
      inst = Builder.CreateICmpULE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::GeSVecI8x16:
      LOG("   op = i8x16.ge_s");
      inst = Builder.CreateICmpSGE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::GeUVecI8x16:
      LOG("   op = i8x16.ge_u");
      inst = Builder.CreateICmpUGE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::EqVecI16x8:
      LOG("   op = i16x8.eq");
      inst = Builder.CreateICmpEQ(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::NeVecI16x8:
      LOG("   op = i16x8.ne");
      inst = Builder.CreateICmpNE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::LtSVecI16x8:
      LOG("   op = i16x8.lt_s");
      inst = Builder.CreateICmpSLT(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::LtUVecI16x8:
      LOG("   op = i16x8.lt_u");
      inst = Builder.CreateICmpULT(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::GtSVecI16x8:
      LOG("   op = i16x8.gt_s");
      inst = Builder.CreateICmpSGT(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::GtUVecI16x8:
      LOG("   op = i16x8.gt_u");
      inst = Builder.CreateICmpUGT(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::LeSVecI16x8:
      LOG("   op = i16x8.le_s");
      inst = Builder.CreateICmpSLE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::LeUVecI16x8:
      LOG("   op = i16x8.le_u");
      inst = Builder.CreateICmpULE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::GeSVecI16x8:
      LOG("   op = i16x8.ge_s");
      inst = Builder.CreateICmpSGE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::GeUVecI16x8:
      LOG("   op = i16x8.ge_u");
      inst = Builder.CreateICmpUGE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::EqVecI32x4:
      LOG("   op = i32x4.eq");
      inst = Builder.CreateICmpEQ(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::NeVecI32x4:
      LOG("   op = i32x4.ne");
      inst = Builder.CreateICmpNE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::LtSVecI32x4:
      LOG("   op = i32x4.lt_s");
      inst = Builder.CreateICmpSLT(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::LtUVecI32x4:
      LOG("   op = i32x4.lt_u");
      inst = Builder.CreateICmpULT(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::GtSVecI32x4:
      LOG("   op = i32x4.gt_s");
      inst = Builder.CreateICmpSGT(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::GtUVecI32x4:
      LOG("   op = i32x4.gt_u");
      inst = Builder.CreateICmpUGT(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::LeSVecI32x4:
      LOG("   op = i32x4.le_s");
      inst = Builder.CreateICmpSLE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::LeUVecI32x4:
      LOG("   op = i32x4.le_u");
      inst = Builder.CreateICmpULE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::GeSVecI32x4:
      LOG("   op = i32x4.ge_s");
      inst = Builder.CreateICmpSGE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::GeUVecI32x4:
      LOG("   op = i32x4.ge_u");
      inst = Builder.CreateICmpUGE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::EqVecI64x2:
      LOG("   op = i64x2.eq");
      inst = Builder.CreateICmpEQ(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::NeVecI64x2:
      LOG("   op = i64x2.ne");
      inst = Builder.CreateICmpNE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::LtSVecI64x2:
      LOG("   op = i64x2.lt_s");
      inst = Builder.CreateICmpSLT(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::GtSVecI64x2:
      LOG("   op = i64x2.gt_s");
      inst = Builder.CreateICmpSGT(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::LeSVecI64x2:
      LOG("   op = i64x2.le_s");
      inst = Builder.CreateICmpSLE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::GeSVecI64x2:
      LOG("   op = i64x2.ge_s");
      inst = Builder.CreateICmpSGE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::EqVecF32x4:
      LOG("   op = f32x4.eq");
      inst = Builder.CreateFCmpOEQ(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::NeVecF32x4:
      LOG("   op = f32x4.ne");
      inst = Builder.CreateFCmpONE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::LtVecF32x4:
      LOG("   op = f32x4.lt");
      inst = Builder.CreateFCmpOLT(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::GtVecF32x4:
      LOG("   op = f32x4.gt");
      inst = Builder.CreateFCmpOGT(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::LeVecF32x4:
      LOG("   op = f32x4.le");
      inst = Builder.CreateFCmpOLE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::GeVecF32x4:
      LOG("   op = f32x4.ge");
      inst = Builder.CreateFCmpOGE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::EqVecF64x2:
      LOG("   op = f64x2.eq");
      inst = Builder.CreateFCmpOEQ(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::NeVecF64x2:
      LOG("   op = f64x2.ne");
      inst = Builder.CreateFCmpONE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::LtVecF64x2:
      LOG("   op = f64x2.lt");
      inst = Builder.CreateFCmpOLT(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::GtVecF64x2:
      LOG("   op = f64x2.gt");
      inst = Builder.CreateFCmpOGT(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::LeVecF64x2:
      LOG("   op = f64x2.le");
      inst = Builder.CreateFCmpOLE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;
    case wasm::GeVecF64x2:
      LOG("   op = f64x2.ge");
      inst = Builder.CreateFCmpOGE(left, right);
      inst = Builder.CreateZExt(inst, Parent.v128);
      break;

    case wasm::AndVec128:
      LOG("   op = v128.and");
      inst = Builder.CreateAnd(left, right);
      break;
    case wasm::OrVec128:
      LOG("   op = v128.or");
      inst = Builder.CreateOr(left, right);
      break;
    case wasm::XorVec128:
      LOG("   op = v128.xor");
      inst = Builder.CreateXor(left, right);
      break;
    case wasm::AndNotVec128:
      LOG("   op = v128.andnot");
      inst = Builder.CreateAnd(left, Builder.CreateNot(right));
      break;

    case wasm::AddVecI8x16:
      LOG("   op = i8x16.add");
      inst = Builder.CreateAdd(left, right);
      break;
    case wasm::AddSatSVecI8x16:
      PANIC("   op = i8x16.add_sat_s");
      break;
    case wasm::AddSatUVecI8x16:
      PANIC("   op = i8x16.add_sat_u");
      break;
    case wasm::SubVecI8x16:
      LOG("   op = i8x16.sub");
      inst = Builder.CreateSub(left, right);
      break;
    case wasm::SubSatSVecI8x16:
      PANIC("   op = i8x16.sub_sat_s");
      break;
    case wasm::SubSatUVecI8x16:
      PANIC("   op = i8x16.sub_sat_u");
      break;
    case wasm::MinSVecI8x16:
      PANIC("   op = i8x16.min_s");
      break;
    case wasm::MinUVecI8x16:
      PANIC("   op = i8x16.min_u");
      break;
    case wasm::MaxSVecI8x16:
      PANIC("   op = i8x16.max_s");
      break;
    case wasm::MaxUVecI8x16:
      PANIC("   op = i8x16.max_u");
      break;
    case wasm::AvgrUVecI8x16:
      PANIC("   op = i8x16.avgr_u");
      break;
    case wasm::AddVecI16x8:
      LOG("   op = i16x8.add");
      inst = Builder.CreateAdd(left, right);
      break;
    case wasm::AddSatSVecI16x8:
      PANIC("   op = i16x8.add_sat_s");
      break;
    case wasm::AddSatUVecI16x8:
      PANIC("   op = i16x8.add_sat_u");
      break;
    case wasm::SubVecI16x8:
      LOG("   op = i16x8.sub");
      inst = Builder.CreateSub(left, right);
      break;
    case wasm::SubSatSVecI16x8:
      PANIC("   op = i16x8.sub_sat_s");
      break;
    case wasm::SubSatUVecI16x8:
      PANIC("   op = i16x8.sub_sat_u");
      break;
    case wasm::MulVecI16x8:
      LOG("   op = i16x8.mul");
      inst = Builder.CreateMul(left, right);
      break;
    case wasm::MinSVecI16x8:
      PANIC("   op = i16x8.min_s");
      break;
    case wasm::MinUVecI16x8:
      PANIC("   op = i16x8.min_u");
      break;
    case wasm::MaxSVecI16x8:
      PANIC("   op = i16x8.max_s");
      break;
    case wasm::MaxUVecI16x8:
      PANIC("   op = i16x8.max_u");
      break;
    case wasm::AvgrUVecI16x8:
      PANIC("   op = i16x8.avgr_u");
      break;
    case wasm::Q15MulrSatSVecI16x8:
      PANIC("   op = i16x8.q15mulr_sat_s");
      break;
    case wasm::ExtMulLowSVecI16x8:
      PANIC("   op = i16x8.extmul_low_i8x16_s");
      break;
    case wasm::ExtMulHighSVecI16x8:
      PANIC("   op = i16x8.extmul_high_i8x16_s");
      break;
    case wasm::ExtMulLowUVecI16x8:
      PANIC("   op = i16x8.extmul_low_i8x16_u");
      break;
    case wasm::ExtMulHighUVecI16x8:
      PANIC("   op = i16x8.extmul_high_i8x16_u");
      break;

    case wasm::AddVecI32x4:
      LOG("   op = i32x4.add");
      inst = Builder.CreateAdd(left, right);
      break;
    case wasm::SubVecI32x4:
      LOG("   op = i32x4.sub");
      inst = Builder.CreateSub(left, right);
      break;
    case wasm::MulVecI32x4:
      LOG("   op = i32x4.mul");
      inst = Builder.CreateMul(left, right);
      break;
    case wasm::MinSVecI32x4:
      PANIC("   op = i32x4.min_s");
      break;
    case wasm::MinUVecI32x4:
      PANIC("   op = i32x4.min_u");
      break;
    case wasm::MaxSVecI32x4:
      PANIC("   op = i32x4.max_s");
      break;
    case wasm::MaxUVecI32x4:
      PANIC("   op = i32x4.max_u");
      break;
    case wasm::DotSVecI16x8ToVecI32x4:
      PANIC("   op = i32x4.dot_i16x8_s");
      break;
    case wasm::ExtMulLowSVecI32x4:
      PANIC("   op = i32x4.extmul_low_i16x8_s");
      break;
    case wasm::ExtMulHighSVecI32x4:
      PANIC("   op = i32x4.extmul_high_i16x8_s");
      break;
    case wasm::ExtMulLowUVecI32x4:
      PANIC("   op = i32x4.extmul_low_i16x8_u");
      break;
    case wasm::ExtMulHighUVecI32x4:
      PANIC("   op = i32x4.extmul_high_i16x8_u");
      break;

    case wasm::AddVecI64x2:
      LOG("   op = i64x2.add");
      inst = Builder.CreateAdd(left, right);
      break;
    case wasm::SubVecI64x2:
      LOG("   op = i64x2.sub");
      inst = Builder.CreateSub(left, right);
      break;
    case wasm::MulVecI64x2:
      LOG("   op = i64x2.mul");
      inst = Builder.CreateMul(left, right);
      break;
    case wasm::ExtMulLowSVecI64x2:
      PANIC("   op = i64x2.extmul_low_i32x4_s");
      break;
    case wasm::ExtMulHighSVecI64x2:
      PANIC("   op = i64x2.extmul_high_i32x4_s");
      break;
    case wasm::ExtMulLowUVecI64x2:
      PANIC("   op = i64x2.extmul_low_i32x4_u");
      break;
    case wasm::ExtMulHighUVecI64x2:
      PANIC("   op = i64x2.extmul_high_i32x4_u");
      break;

    case wasm::AddVecF32x4:
      LOG("   op = f32x4.add");
      inst = Builder.CreateFAdd(left, right);
      break;
    case wasm::SubVecF32x4:
      LOG("   op = f32x4.sub");
      inst = Builder.CreateFSub(left, right);
      break;
    case wasm::MulVecF32x4:
      LOG("   op = f32x4.mul");
      inst = Builder.CreateFMul(left, right);
      break;
    case wasm::DivVecF32x4:
      LOG("   op = f32x4.div");
      inst = Builder.CreateFDiv(left, right);
      break;
    case wasm::MinVecF32x4:
      LOG("   op = f32x4.min");
      inst = Builder.CreateMinimum(left, right);
      break;
    case wasm::MaxVecF32x4:
      LOG("   op = f32x4.max");
      inst = Builder.CreateMaximum(left, right);
      break;
    case wasm::PMinVecF32x4:
      LOG("   op = f32x4.pmin");
      inst = Builder.CreateMinNum(left, right);
      break;
    case wasm::PMaxVecF32x4:
      LOG("   op = f32x4.pmax");
      inst = Builder.CreateMaxNum(left, right);
      break;
    case wasm::AddVecF64x2:
      LOG("   op = f64x2.add");
      inst = Builder.CreateFAdd(left, right);
      break;
    case wasm::SubVecF64x2:
      LOG("   op = f64x2.sub");
      inst = Builder.CreateFSub(left, right);
      break;
    case wasm::MulVecF64x2:
      LOG("   op = f64x2.mul");
      inst = Builder.CreateFMul(left, right);
      break;
    case wasm::DivVecF64x2:
      LOG("   op = f64x2.div");
      inst = Builder.CreateFDiv(left, right);
      break;
    case wasm::MinVecF64x2:
      LOG("   op = f64x2.min");
      inst = Builder.CreateMinimum(left, right);
      break;
    case wasm::MaxVecF64x2:
      LOG("   op = f64x2.max");
      inst = Builder.CreateMaximum(left, right);
      break;
    case wasm::PMinVecF64x2:
      LOG("   op = f64x2.pmin");
      inst = Builder.CreateMinNum(left, right);
      break;
    case wasm::PMaxVecF64x2:
      LOG("   op = f64x2.pmax");
      inst = Builder.CreateMaxNum(left, right);
      break;

    case wasm::NarrowSVecI16x8ToVecI8x16:
      PANIC("   op = i8x16.narrow_i16x8_s");
      break;
    case wasm::NarrowUVecI16x8ToVecI8x16:
      PANIC("   op = i8x16.narrow_i16x8_u");
      break;
    case wasm::NarrowSVecI32x4ToVecI16x8:
      PANIC("   op = i16x8.narrow_i32x4_s");
      break;
    case wasm::NarrowUVecI32x4ToVecI16x8:
      PANIC("   op = i16x8.narrow_i32x4_u");
      break;

    case wasm::SwizzleVecI8x16:
      PANIC("   op = i8x16.swizzle");
      break;

    case wasm::RelaxedMinVecF32x4:
      PANIC("   op = f32x4.relaxed_min");
      break;
    case wasm::RelaxedMaxVecF32x4:
      PANIC("   op = f32x4.relaxed_max");
      break;
    case wasm::RelaxedMinVecF64x2:
      PANIC("   op = f64x2.relaxed_min");
      break;
    case wasm::RelaxedMaxVecF64x2:
      PANIC("   op = f64x2.relaxed_max");
      break;
    case wasm::RelaxedSwizzleVecI8x16:
      PANIC("   op = i8x16.relaxed_swizzle");
      break;
    case wasm::RelaxedQ15MulrSVecI16x8:
      LOG("   op = i16x8.relaxed_q15mulr_s");
      break;
    case wasm::DotI8x16I7x16SToVecI16x8:
      PANIC("   op = i16x8.dot_i8x16_i7x16_s");
      break;

    case wasm::InvalidBinary:
      PANIC("unvalid binary operator");
  }

  return inst;
}

Value* LLVMPass::IREmitter::visitGlobalGet(wasm::GlobalGet* curr) {
  Value* inst = nullptr;
  auto name = curr->name.toString();

  LOG("  ----> IREmitter::visitGlobalGet: " + name);

  auto global = Module.getGlobalVariable(name, true);
  auto ty = Parent.wasmTypeToLLVM(curr->type);
  inst = Builder.CreateLoad(ty, global);

  return inst;
}

Value* LLVMPass::IREmitter::assureBool(Value* val) {
  Value* condition = val;
  if (condition != nullptr) {
    auto cond_ty = condition->getType();

    if (!cond_ty->isIntegerTy()) {
      PANIC("Select condition is not integer");
    }
    int bits = cond_ty->getIntegerBitWidth();

    if (bits != 1) {
      condition = eqz(condition);
    }
  }

  return condition;
}

Value* LLVMPass::IREmitter::visitSelect(wasm::Select* curr) {
  LOG("  ----> IREmitter::visitSelect: ");

  Value* inst = nullptr;

  auto ifTrue = visit(curr->ifTrue);
  auto ifFalse = visit(curr->ifFalse);
  auto condition = visit(curr->condition);

  // LOG(" .  Cond:"); DUMP(condition);
  // LOG(" .   True: "); DUMP(ifTrue);
  // LOG(" .    False: "); DUMP(ifFalse);

  inst = Builder.CreateSelect(assureBool(condition), ifTrue, ifFalse);

  return inst;
}

Value* LLVMPass::IREmitter::visitUnary(wasm::Unary* curr) {
  LOG("  ----> IREmitter::visitUnary: ");

  Value* inst = nullptr;

  auto val = visit(curr->value);

  switch (curr->op) {
    case wasm::ClzInt32:
      LOG(".   op = i32.clz");
      assureType(val, Parent.i32);
      inst = Builder.CreateBinaryIntrinsic(
        Intrinsic::ctlz, val, ConstantInt::get(Parent.i1, 0));
      break;
    case wasm::CtzInt32:
      LOG(".   op = i32.ctz");
      assureType(val, Parent.i32);
      inst = Builder.CreateBinaryIntrinsic(
        Intrinsic::cttz, val, ConstantInt::get(Parent.i1, 0));
      break;
    case wasm::PopcntInt32:
      LOG(".   op = i32.popcnt");
      assureType(val, Parent.i32);
      inst = Builder.CreateUnaryIntrinsic(Intrinsic::ctpop, val);
      break;
    case wasm::EqZInt32:
      LOG(".   op = i32.eqz");
      assureType(val, Parent.i32);
      inst = this->eqz(val);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::ClzInt64:
      LOG(".   op = i64.clz");
      assureType(val, Parent.i64);
      inst = Builder.CreateBinaryIntrinsic(
        Intrinsic::ctlz, val, ConstantInt::get(Parent.i1, 0));
      break;
    case wasm::CtzInt64:
      LOG(".   op = i64.ctz");
      assureType(val, Parent.i64);
      inst = Builder.CreateBinaryIntrinsic(
        Intrinsic::cttz, val, ConstantInt::get(Parent.i1, 0));
      break;
    case wasm::PopcntInt64:
      LOG(".   op = i64.popcnt");
      assureType(val, Parent.i64);
      inst = Builder.CreateUnaryIntrinsic(Intrinsic::ctpop, val);
      break;
    case wasm::EqZInt64:
      LOG(".   op = i64.eqz");
      assureType(val, Parent.i64);
      inst = this->eqz(val);
      inst = Builder.CreateZExt(inst, Parent.i32);
      break;
    case wasm::NegFloat32:
      LOG(".   op = f32.neg");
      assureType(val, Parent.f32);
      inst = Builder.CreateFNeg(val);
      break;
    case wasm::AbsFloat32:
      LOG(".   op = f32.abs");
      assureType(val, Parent.f32);
      inst = Builder.CreateUnaryIntrinsic(Intrinsic::fabs, val);
      break;
    case wasm::CeilFloat32:
      LOG(".   op = f32.ceil");
      assureType(val, Parent.f32);
      inst = Builder.CreateUnaryIntrinsic(Intrinsic::ceil, val);
      break;
    case wasm::FloorFloat32:
      LOG(".   op = f32.floor");
      assureType(val, Parent.f32);
      inst = Builder.CreateUnaryIntrinsic(Intrinsic::floor, val);
      break;
    case wasm::TruncFloat32:
      LOG(".   op = f32.trunc");
      assureType(val, Parent.f32);
      inst = Builder.CreateUnaryIntrinsic(Intrinsic::trunc, val);
      break;
    case wasm::NearestFloat32:
      LOG(".   op = f32.nearest");
      assureType(val, Parent.f32);
      inst = Builder.CreateUnaryIntrinsic(Intrinsic::rint, val);
      break;
    case wasm::SqrtFloat32:
      LOG(".   op = f32.sqrt");
      assureType(val, Parent.f32);
      inst = Builder.CreateUnaryIntrinsic(Intrinsic::sqrt, val);
      break;
    case wasm::NegFloat64:
      LOG(".   op = f64.neg");
      assureType(val, Parent.f64);
      inst = Builder.CreateFNeg(val);
      break;
    case wasm::AbsFloat64:
      LOG(".   op = f64.abs");
      assureType(val, Parent.f64);
      inst = Builder.CreateUnaryIntrinsic(Intrinsic::fabs, val);
      break;
    case wasm::CeilFloat64:
      LOG(".   op = f64.ceil");
      assureType(val, Parent.f64);
      inst = Builder.CreateUnaryIntrinsic(Intrinsic::ceil, val);
      break;
    case wasm::FloorFloat64:
      LOG(".   op = f64.floor");
      assureType(val, Parent.f64);
      inst = Builder.CreateUnaryIntrinsic(Intrinsic::floor, val);
      break;
    case wasm::TruncFloat64:
      LOG(".   op = f64.trunc");
      assureType(val, Parent.f64);
      inst = Builder.CreateUnaryIntrinsic(Intrinsic::trunc, val);
      break;
    case wasm::NearestFloat64:
      LOG(".   op = f64.nearest");
      assureType(val, Parent.f64);
      inst = Builder.CreateUnaryIntrinsic(Intrinsic::rint, val);
      break;
    case wasm::SqrtFloat64:
      LOG(".   op = f64.sqrt");
      assureType(val, Parent.f64);
      inst = Builder.CreateUnaryIntrinsic(Intrinsic::sqrt, val);
      break;
    case wasm::ExtendSInt32:
      LOG(".   op = i64.extend_i32_s");
      assureType(val, Parent.i32);
      inst = Builder.CreateSExt(val, Parent.i64);
      break;
    case wasm::ExtendUInt32:
      LOG(".   op = i64.extend_i32_u");
      assureType(val, Parent.i32);
      inst = Builder.CreateZExt(val, Parent.i64);
      break;
    case wasm::WrapInt64:
      LOG(".   op = i32.wrap_i64");
      assureType(val, Parent.i64);
      inst = Builder.CreateTrunc(val, Parent.i32);
      break;
    case wasm::TruncSFloat32ToInt32:
      LOG(".   op = i32.trunc_f32_s");
      assureType(val, Parent.f32);
      inst = Builder.CreateFPToSI(val, Parent.i32);
      break;
    case wasm::TruncSFloat32ToInt64:
      LOG(".   op = i64.trunc_f32_s");
      assureType(val, Parent.f32);
      inst = Builder.CreateFPToSI(val, Parent.i64);
      break;
    case wasm::TruncUFloat32ToInt32:
      LOG(".   op = i32.trunc_f32_u");
      assureType(val, Parent.f32);
      inst = Builder.CreateFPToUI(val, Parent.i32);
      break;
    case wasm::TruncUFloat32ToInt64:
      LOG(".   op = i64.trunc_f32_u");
      assureType(val, Parent.f32);
      inst = Builder.CreateFPToUI(val, Parent.i64);
      break;
    case wasm::TruncSFloat64ToInt32:
      LOG(".   op = i32.trunc_f64_s");
      assureType(val, Parent.f64);
      inst = Builder.CreateFPToSI(val, Parent.i32);
      break;
    case wasm::TruncSFloat64ToInt64:
      LOG(".   op = i64.trunc_f64_s");
      assureType(val, Parent.f64);
      inst = Builder.CreateFPToSI(val, Parent.i64);
      break;
    case wasm::TruncUFloat64ToInt32:
      LOG(".   op = i32.trunc_f64_u");
      assureType(val, Parent.f64);
      inst = Builder.CreateFPToUI(val, Parent.i32);
      break;
    case wasm::TruncUFloat64ToInt64:
      LOG(".   op = i64.trunc_f64_u");
      assureType(val, Parent.f64);
      inst = Builder.CreateFPToUI(val, Parent.i64);
      break;
    case wasm::ReinterpretFloat32:
      LOG(".   op = i32.reinterpret_f32");
      assureType(val, Parent.f32);
      inst = Builder.CreateBitCast(val, Parent.i32);
      break;
    case wasm::ReinterpretFloat64:
      LOG(".   op = i64.reinterpret_f64");
      assureType(val, Parent.f64);
      inst = Builder.CreateBitCast(val, Parent.i64);
      break;
    case wasm::ConvertUInt32ToFloat32:
      LOG(".   op = f32.convert_i32_u");
      assureType(val, Parent.i32);
      inst = Builder.CreateUIToFP(val, Parent.f32);
      break;
    case wasm::ConvertUInt32ToFloat64:
      LOG(".   op = f64.convert_i32_u");
      assureType(val, Parent.i32);
      inst = Builder.CreateUIToFP(val, Parent.f64);
      break;
    case wasm::ConvertSInt32ToFloat32:
      LOG(".   op = f32.convert_i32_s");
      assureType(val, Parent.i32);
      inst = Builder.CreateSIToFP(val, Parent.f32);
      break;
    case wasm::ConvertSInt32ToFloat64:
      LOG(".   op = f64.convert_i32_s");
      assureType(val, Parent.i32);
      inst = Builder.CreateSIToFP(val, Parent.f64);
      break;
    case wasm::ConvertUInt64ToFloat32:
      LOG(".   op = f32.convert_i64_u");
      assureType(val, Parent.i64);
      inst = Builder.CreateUIToFP(val, Parent.f32);
      break;
    case wasm::ConvertUInt64ToFloat64:
      LOG(".   op = f64.convert_i64_u");
      assureType(val, Parent.i64);
      inst = Builder.CreateUIToFP(val, Parent.f64);
      break;
    case wasm::ConvertSInt64ToFloat32:
      LOG(".   op = f32.convert_i64_s");
      assureType(val, Parent.i64);
      inst = Builder.CreateSIToFP(val, Parent.f32);
      break;
    case wasm::ConvertSInt64ToFloat64:
      LOG(".   op = f64.convert_i64_s");
      assureType(val, Parent.i64);
      inst = Builder.CreateSIToFP(val, Parent.f64);
      break;
    case wasm::PromoteFloat32:
      LOG(".   op = f64.promote_f32");
      assureType(val, Parent.f32);
      inst = Builder.CreateFPExt(val, Parent.f64);
      break;
    case wasm::DemoteFloat64:
      LOG(".   op = f32.demote_f64");
      assureType(val, Parent.f64);
      inst = Builder.CreateFPTrunc(val, Parent.f32);
      break;
    case wasm::ReinterpretInt32:
      LOG(".   op = f32.reinterpret_i32");
      assureType(val, Parent.i32);
      inst = Builder.CreateBitCast(val, Parent.f32);
      break;
    case wasm::ReinterpretInt64:
      LOG(".   op = f64.reinterpret_i64");
      assureType(val, Parent.i64);
      inst = Builder.CreateBitCast(val, Parent.f64);
      break;
    case wasm::ExtendS8Int32:
      LOG(".   op = i32.extend8_s");
      assureType(val, Parent.i32);
      val = Builder.CreateSRem(val, ConstantInt::get(Parent.i32, 256));
      inst = Builder.CreateSExt(val, Parent.i32);
      break;
    case wasm::ExtendS16Int32:
      LOG(".   op = i32.extend16_s");
      assureType(val, Parent.i32);
      val = Builder.CreateSRem(val, ConstantInt::get(Parent.i32, 65536));
      inst = Builder.CreateSExt(val, Parent.i32);
      break;
    case wasm::ExtendS8Int64:
      LOG(".   op = i64.extend8_s");
      assureType(val, Parent.i64);
      val = Builder.CreateSRem(val, ConstantInt::get(Parent.i64, 256));
      inst = Builder.CreateSExt(val, Parent.i64);
      break;
    case wasm::ExtendS16Int64:
      LOG(".   op = i64.extend16_s");
      assureType(val, Parent.i64);
      val = Builder.CreateSRem(val, ConstantInt::get(Parent.i64, 65536));
      inst = Builder.CreateSExt(val, Parent.i64);
      break;
    case wasm::ExtendS32Int64:
      LOG(".   op = i64.extend32_s");
      assureType(val, Parent.i64);
      val = Builder.CreateSRem(val, ConstantInt::get(Parent.i64, 1ull << 32));
      inst = Builder.CreateSExt(val, Parent.i64);
      break;
    case wasm::TruncSatSFloat32ToInt32:
      LOG(".   op = i32.trunc_sat_f32_s");
      assureType(val, Parent.f32);
      inst = Builder.CreateIntrinsic(
        Intrinsic::fptosi_sat, {Parent.i32, Parent.f32}, {val});
      break;
    case wasm::TruncSatUFloat32ToInt32:
      LOG(".   op = i32.trunc_sat_f32_u");
      assureType(val, Parent.f32);
      inst = Builder.CreateIntrinsic(
        Intrinsic::fptoui_sat, {Parent.i32, Parent.f32}, {val});
      break;
    case wasm::TruncSatSFloat64ToInt32:
      LOG(".   op = i32.trunc_sat_f64_s");
      assureType(val, Parent.f64);
      inst = Builder.CreateIntrinsic(
        Intrinsic::fptosi_sat, {Parent.i32, Parent.f64}, {val});
      break;
    case wasm::TruncSatUFloat64ToInt32:
      LOG(".   op = i32.trunc_sat_f64_u");
      assureType(val, Parent.f64);
      inst = Builder.CreateIntrinsic(
        Intrinsic::fptoui_sat, {Parent.i32, Parent.f64}, {val});
      break;
    case wasm::TruncSatSFloat32ToInt64:
      LOG(".   op = i64.trunc_sat_f32_s");
      assureType(val, Parent.f32);
      inst = Builder.CreateIntrinsic(
        Intrinsic::fptosi_sat, {Parent.i64, Parent.f32}, {val});
      break;
    case wasm::TruncSatUFloat32ToInt64:
      LOG(".   op = i64.trunc_sat_f32_u");
      assureType(val, Parent.f32);
      inst = Builder.CreateIntrinsic(
        Intrinsic::fptoui_sat, {Parent.i64, Parent.f32}, {val});
      break;
    case wasm::TruncSatSFloat64ToInt64:
      LOG(".   op = i64.trunc_sat_f64_s");
      assureType(val, Parent.f64);
      inst = Builder.CreateIntrinsic(
        Intrinsic::fptosi_sat, {Parent.i64, Parent.f64}, {val});
      break;
    case wasm::TruncSatUFloat64ToInt64:
      LOG(".   op = i64.trunc_sat_f64_u");
      assureType(val, Parent.f64);
      inst = Builder.CreateIntrinsic(
        Intrinsic::fptoui_sat, {Parent.i64, Parent.f64}, {val});
      break;
    case wasm::SplatVecI8x16:
      LOG(".   op = i8x16.splat");
      break;
    case wasm::SplatVecI16x8:
      LOG(".   op = i16x8.splat");
      break;
    case wasm::SplatVecI32x4:
      LOG(".   op = i32x4.splat");
      break;
    case wasm::SplatVecI64x2:
      LOG(".   op = i64x2.splat");
      break;
    case wasm::SplatVecF32x4:
      LOG(".   op = f32x4.splat");
      break;
    case wasm::SplatVecF64x2:
      LOG(".   op = f64x2.splat");
      break;
    case wasm::NotVec128:
      LOG(".   op = v128.not");
      break;
    case wasm::AnyTrueVec128:
      LOG(".   op = v128.any_true");
      break;
    case wasm::AbsVecI8x16:
      LOG(".   op = i8x16.abs");
      break;
    case wasm::NegVecI8x16:
      LOG(".   op = i8x16.neg");
      break;
    case wasm::AllTrueVecI8x16:
      LOG(".   op = i8x16.all_true");
      break;
    case wasm::BitmaskVecI8x16:
      LOG(".   op = i8x16.bitmask");
      break;
    case wasm::PopcntVecI8x16:
      LOG(".   op = i8x16.popcnt");
      break;
    case wasm::AbsVecI16x8:
      LOG(".   op = i16x8.abs");
      break;
    case wasm::NegVecI16x8:
      LOG(".   op = i16x8.neg");
      break;
    case wasm::AllTrueVecI16x8:
      LOG(".   op = i16x8.all_true");
      break;
    case wasm::BitmaskVecI16x8:
      LOG(".   op = i16x8.bitmask");
      break;
    case wasm::AbsVecI32x4:
      LOG(".   op = i32x4.abs");
      break;
    case wasm::NegVecI32x4:
      LOG(".   op = i32x4.neg");
      break;
    case wasm::AllTrueVecI32x4:
      LOG(".   op = i32x4.all_true");
      break;
    case wasm::BitmaskVecI32x4:
      LOG(".   op = i32x4.bitmask");
      break;
    case wasm::AbsVecI64x2:
      LOG(".   op = i64x2.abs");
      break;
    case wasm::NegVecI64x2:
      LOG(".   op = i64x2.neg");
      break;
    case wasm::AllTrueVecI64x2:
      LOG(".   op = i64x2.all_true");
      break;
    case wasm::BitmaskVecI64x2:
      LOG(".   op = i64x2.bitmask");
      break;
    case wasm::AbsVecF32x4:
      LOG(".   op = f32x4.abs");
      break;
    case wasm::NegVecF32x4:
      LOG(".   op = f32x4.neg");
      break;
    case wasm::SqrtVecF32x4:
      LOG(".   op = f32x4.sqrt");
      break;
    case wasm::CeilVecF32x4:
      LOG(".   op = f32x4.ceil");
      break;
    case wasm::FloorVecF32x4:
      LOG(".   op = f32x4.floor");
      break;
    case wasm::TruncVecF32x4:
      LOG(".   op = f32x4.trunc");
      break;
    case wasm::NearestVecF32x4:
      LOG(".   op = f32x4.nearest");
      break;
    case wasm::AbsVecF64x2:
      LOG(".   op = f64x2.abs");
      break;
    case wasm::NegVecF64x2:
      LOG(".   op = f64x2.neg");
      break;
    case wasm::SqrtVecF64x2:
      LOG(".   op = f64x2.sqrt");
      break;
    case wasm::CeilVecF64x2:
      LOG(".   op = f64x2.ceil");
      break;
    case wasm::FloorVecF64x2:
      LOG(".   op = f64x2.floor");
      break;
    case wasm::TruncVecF64x2:
      LOG(".   op = f64x2.trunc");
      break;
    case wasm::NearestVecF64x2:
      LOG(".   op = f64x2.nearest");
      break;
    case wasm::ExtAddPairwiseSVecI8x16ToI16x8:
      LOG(".   op = i16x8.extadd_pairwise_i8x16_s");
      break;
    case wasm::ExtAddPairwiseUVecI8x16ToI16x8:
      LOG(".   op = i16x8.extadd_pairwise_i8x16_u");
      break;
    case wasm::ExtAddPairwiseSVecI16x8ToI32x4:
      LOG(".   op = i32x4.extadd_pairwise_i16x8_s");
      break;
    case wasm::ExtAddPairwiseUVecI16x8ToI32x4:
      LOG(".   op = i32x4.extadd_pairwise_i16x8_u");
      break;
    case wasm::TruncSatSVecF32x4ToVecI32x4:
      LOG(".   op = i32x4.trunc_sat_f32x4_s");
      break;
    case wasm::TruncSatUVecF32x4ToVecI32x4:
      LOG(".   op = i32x4.trunc_sat_f32x4_u");
      break;
    case wasm::ConvertSVecI32x4ToVecF32x4:
      LOG(".   op = f32x4.convert_i32x4_s");
      break;
    case wasm::ConvertUVecI32x4ToVecF32x4:
      LOG(".   op = f32x4.convert_i32x4_u");
      break;
    case wasm::ExtendLowSVecI8x16ToVecI16x8:
      LOG(".   op = i16x8.extend_low_i8x16_s");
      break;
    case wasm::ExtendHighSVecI8x16ToVecI16x8:
      LOG(".   op = i16x8.extend_high_i8x16_s");
      break;
    case wasm::ExtendLowUVecI8x16ToVecI16x8:
      LOG(".   op = i16x8.extend_low_i8x16_u");
      break;
    case wasm::ExtendHighUVecI8x16ToVecI16x8:
      LOG(".   op = i16x8.extend_high_i8x16_u");
      break;
    case wasm::ExtendLowSVecI16x8ToVecI32x4:
      LOG(".   op = i32x4.extend_low_i16x8_s");
      break;
    case wasm::ExtendHighSVecI16x8ToVecI32x4:
      LOG(".   op = i32x4.extend_high_i16x8_s");
      break;
    case wasm::ExtendLowUVecI16x8ToVecI32x4:
      LOG(".   op = i32x4.extend_low_i16x8_u");
      break;
    case wasm::ExtendHighUVecI16x8ToVecI32x4:
      LOG(".   op = i32x4.extend_high_i16x8_u");
      break;
    case wasm::ExtendLowSVecI32x4ToVecI64x2:
      LOG(".   op = i64x2.extend_low_i32x4_s");
      break;
    case wasm::ExtendHighSVecI32x4ToVecI64x2:
      LOG(".   op = i64x2.extend_high_i32x4_s");
      break;
    case wasm::ExtendLowUVecI32x4ToVecI64x2:
      LOG(".   op = i64x2.extend_low_i32x4_u");
      break;
    case wasm::ExtendHighUVecI32x4ToVecI64x2:
      LOG(".   op = i64x2.extend_high_i32x4_u");
      break;
    case wasm::ConvertLowSVecI32x4ToVecF64x2:
      LOG(".   op = f64x2.convert_low_i32x4_s");
      break;
    case wasm::ConvertLowUVecI32x4ToVecF64x2:
      LOG(".   op = f64x2.convert_low_i32x4_u");
      break;
    case wasm::TruncSatZeroSVecF64x2ToVecI32x4:
      LOG(".   op = i32x4.trunc_sat_f64x2_s_zero");
      break;
    case wasm::TruncSatZeroUVecF64x2ToVecI32x4:
      LOG(".   op = i32x4.trunc_sat_f64x2_u_zero");
      break;
    case wasm::DemoteZeroVecF64x2ToVecF32x4:
      LOG(".   op = f32x4.demote_f64x2_zero");
      break;
    case wasm::PromoteLowVecF32x4ToVecF64x2:
      LOG(".   op = f64x2.promote_low_f32x4");
      break;
    case wasm::RelaxedTruncSVecF32x4ToVecI32x4:
      LOG(".   op = i32x4.relaxed_trunc_f32x4_s");
      break;
    case wasm::RelaxedTruncUVecF32x4ToVecI32x4:
      LOG(".   op = i32x4.relaxed_trunc_f32x4_u");
      break;
    case wasm::RelaxedTruncZeroSVecF64x2ToVecI32x4:
      LOG(".   op = i32x4.relaxed_trunc_f64x2_s_zero");
      break;
    case wasm::RelaxedTruncZeroUVecF64x2ToVecI32x4:
      LOG(".   op = i32x4.relaxed_trunc_f64x2_u_zero");
      break;
    case wasm::InvalidUnary:
      PANIC("unvalid unary operator");
  }

  return inst;
}

Value* LLVMPass::IREmitter::visitReturn(wasm::Return* curr) {
  LOG("  ----> IREmitter::visitReturn: ");

  Value* value = nullptr;
  if (!curr->value) {
    Builder.CreateRetVoid();
  } else {
    value = visit(curr->value);
    Builder.CreateRet(value);
  }

  return value;
}

Value* LLVMPass::IREmitter::visitGlobalSet(wasm::GlobalSet* curr) {
  Value* inst = nullptr;
  auto name = curr->name.toString();

  LOG("  ----> IREmitter::visitGlobalSet: " + name);

  auto global = Module.getGlobalVariable(name, true);
  auto val = visit(curr->value);
  inst = Builder.CreateStore(val, global);

  return inst;
}

Value* LLVMPass::IREmitter::visitBreak(wasm::Break* curr) {
  LOG("  ----> IREmitter::visitBreak: ");

  auto name = curr->name.toString();
  auto dest = Parent.namedBlocks[name];
  if (!dest || !dest->getType()->isLabelTy()) {
    PANIC(" .  Not a label: " + name);
  }

  LOG("  .    Label: " + name);

  Value* value = nullptr;

  if (curr->value) {
    value = visit(curr->value);
    LOG("  .    Br with value: ");
    DUMP(value);
  }

  if (curr->condition) {
    auto cond = assureBool(visit(curr->condition));
    BasicBlock* next = createBlock("", Func);
    if (value != nullptr) {
      Parent.branchMap[dest].push_back({Builder.GetInsertBlock(), value});
    }
    Builder.CreateCondBr(cond, dest, next);
    Builder.SetInsertPoint(next);
  } else {
    if (value != nullptr) {
      Parent.branchMap[dest].push_back({Builder.GetInsertBlock(), value});
    }
    Builder.CreateBr(dest);
  }

  return value;
}

void LLVMPass::IREmitter::iterateBody(wasm::Expression* body) {
  if (body->is<wasm::Block>() && body->cast<wasm::Block>()->name.isNull()) {
    wasm::Block* block = body->cast<wasm::Block>();
    for (auto item : block->list) {
      this->visit(item);
    }
  } else {
    this->visit(body);
  }
}

Value* LLVMPass::IREmitter::visitLoop(wasm::Loop* curr) {
  LOG("  ----> IREmitter::visitLoop: ");

  auto name = curr->name.toString();
  BasicBlock* prevInstPt = Builder.GetInsertBlock();
  BasicBlock* block = createBlock(name, Func);

  if (prevInstPt->getTerminator() == nullptr) {
    Builder.CreateBr(block);
  }

  if (curr->type.isConcrete()) {
    LOG("   Loop has result type: " + curr->type.toString());
  }

  Builder.SetInsertPoint(block);
  LOG(
    "  ----> IREmitter::visitLoop: " +
    (curr->name.isNull() ? "(null name)" : name) +
    (curr->type.isConcrete() ? " (result " + curr->type.toString() + ")" : ""));

  this->iterateBody(curr->body);

  LOG("  <----- IREmitter::doneWithLoop: " + name);

  if (curr->type.isConcrete()) {
    LOG("   Loop has result type: " + curr->type.toString());

    auto branches = Parent.branchMap[block];
    if (branches.size() > 1) {
      LOG("   Block branch cnt: " + std::to_string(branches.size()));
      auto type = Parent.wasmTypeToLLVM(curr->type);
      auto phi = Builder.CreatePHI(type, branches.size());
      for (auto [from, value] : branches) {
        LOG(" .   adding Branch from : " + from->getNameOrAsOperand());
        DUMP(value);

        phi->addIncoming(value, from);
      }
      return phi;
    } else {
      return Parent.lastValue;
    }
  }

  return block;
}

Value* LLVMPass::IREmitter::visitDrop(wasm::Drop* curr) {
  LOG("  ----> IREmitter::visitDrop: ");

  visit(curr->value);

  return nullptr;
}

Value* LLVMPass::IREmitter::visitUnreachable(wasm::Unreachable* curr) {
  LOG("  ----> IREmitter::visitUnreachable: ");

  return Builder.CreateUnreachable();
}

Value* LLVMPass::IREmitter::visitMemorySize(wasm::MemorySize* curr) {
  LOG("  ----> IREmitter::visitMemorySize: ");

  auto name = Parent.memoryName(curr->memory.toString());
  auto& mem = Parent.namedMemories[name];
  int mem_size = mem.size;
  LOG(" .     Memory: " + name);

  Value* inst =
    ConstantInt::get(Parent.i32, mem_size / wasm::Memory::kPageSize);

  return inst;
}

Value* LLVMPass::IREmitter::visitMemoryCopy(wasm::MemoryCopy* curr) {
  LOG("  ----> IREmitter::visitMemoryCopy: ");

  auto destName = Parent.memoryName(curr->destMemory.toString());
  LOG(" .   Dest Memory: " + destName);
  auto srcName = Parent.memoryName(curr->sourceMemory.toString());
  LOG(" .   Source Memory: " + srcName);

  auto& destMemory = Parent.namedMemories[destName];
  auto& srcMemory = Parent.namedMemories[srcName];

  auto dest = Builder.CreateGEP(
    IntegerType::get(Context, 8), destMemory.ptr, visit(curr->dest));

  auto src = Builder.CreateGEP(
    IntegerType::get(Context, 8), srcMemory.ptr, visit(curr->source));

  auto size = visit(curr->size);
  LOG("       Src:");
  DUMP(src);
  LOG("       Dest:");
  DUMP(dest);
  LOG("       Size:");
  DUMP(size);
  auto inst = Builder.CreateMemCpy(dest, Align(1), src, Align(1), size);
  return inst;
}

Value* LLVMPass::IREmitter::visitSwitch(wasm::Switch* curr) {
  LOG("  ----> IREmitter::visitSwitch: ");

  int ncase = curr->targets.size();
  auto defaultCase = curr->default_.toString();
  BasicBlock* defaultBB = Parent.namedBlocks[defaultCase];

  auto condition = visit(curr->condition);

  auto inst = Builder.CreateSwitch(condition, defaultBB, ncase);

  for (int i = 0; i < ncase; i++) {
    auto caseName = curr->targets[i].toString();
    BasicBlock* caseBB = Parent.namedBlocks[caseName];
    inst->addCase(ConstantInt::get((IntegerType*)Parent.i32, i), caseBB);
  }

  LOG(" .  Switch inst:");
  DUMP(inst);

  return inst;
}

Value* LLVMPass::IREmitter::visitCallIndirect(wasm::CallIndirect* curr) {
  LOG("  ----> IREmitter::visitCallIndirect: ");

  auto tname = Parent.tableName(curr->table.toString());
  auto table = Module.getNamedGlobal(tname);

  if (table == nullptr) {
    LOG(" .    Table not found:" + tname);
  } else {
    LOG(" .   Table: ");
    DUMP(table);
  }

  auto target = visit(curr->target);
  if (target == nullptr) {
    LOG(" .     Target is null");
  } else {
    LOG(" .   Targe: ");
    DUMP(target);
  }

  auto signature = curr->heapType.getSignature();
  auto type = Parent.wasmSignatureToLLVM(signature);
  auto func = Builder.CreateGEP(Parent.funcRef, table, target);

  LOG(" .     Function is ");
  DUMP(func);
  DUMP(type);

  std::vector<Value*> args;
  for (auto it = curr->operands.begin(); it != curr->operands.end(); it++) {
    wasm::Expression* op = *it;
    args.push_back(visit(op));
  }

  auto inst = Builder.CreateCall(type, func, args);
  inst->setTailCall(curr->isReturn);
  LOG(" .     CallInst is ");
  DUMP(inst);
  return inst;
}

Value* LLVMPass::IREmitter::visitRefFunc(wasm::RefFunc* curr) {
  LOG("  ----> IREmitter::visitRefFunc: ");

  auto fname = curr->func.toString();
  LOG(" .     Function Name: " + fname);

  auto func = Module.getFunction(fname);

  if (!func) {
    LOG(" .   Function not found");
    func = assureFunction(fname, {});
  }

  if (!func->getType()->isPointerTy()) {
    LOG(" .  Function is not pointer: ");
  } else {
    LOG(" .    Function Type:");
  }
  DUMP(func->getType());

  return func;
}

Value* LLVMPass::IREmitter::visitIf(wasm::If* curr) {
  LOG("  ----> IREmitter::visitIf: ");

  Value* inst = nullptr;

  auto type = Parent.wasmTypeToLLVM(curr->type);

  BasicBlock* ifBB = createBlock("If");
  BasicBlock* thenBB = createBlock("Then");
  BasicBlock* elseBB = createBlock("Else");
  BasicBlock* endifBB = createBlock("EndIf");

  appendInsertionBlock(ifBB);
  LOG(" .   Resolving Condition : ");
  auto cond = assureBool(visit(curr->condition));
  Builder.CreateCondBr(cond, thenBB, elseBB);

  appendInsertionBlock(thenBB);
  LOG(" .   Resolving True : ");
  auto ifTrue = visit(curr->ifTrue);
  thenBB = Builder.GetInsertBlock();
  if (thenBB->getTerminator() == nullptr) {
    Builder.CreateBr(endifBB);
    if (ifTrue != nullptr) {
      Parent.branchMap[endifBB].push_back({thenBB, ifTrue});
    }
  }

  appendInsertionBlock(elseBB);
  Value* ifFalse = nullptr;
  if (curr->ifFalse) {
    LOG(" .   Resolving false : ");
    ifFalse = visit(curr->ifFalse);
  }

  elseBB = Builder.GetInsertBlock();
  if (elseBB->getTerminator() == nullptr) {
    Builder.CreateBr(endifBB);
    if (ifFalse != nullptr) {
      Parent.branchMap[endifBB].push_back({elseBB, ifFalse});
    }
  }

  appendInsertionBlock(endifBB);
  if (!type->isVoidTy()) {
    auto branches = Parent.branchMap[endifBB];
    LOG("   EndIf branch cnt: " + std::to_string(branches.size()));
    auto phi = Builder.CreatePHI(type, branches.size());
    for (auto [from, value] : branches) {
      LOG(" .   adding Branch from : " + from->getNameOrAsOperand());
      DUMP(value);

      phi->addIncoming(value, from);
    }
    inst = phi;
  }

  return inst;
}

Value* LLVMPass::IREmitter::visitNop(wasm::Nop* curr) {
  LOG("  ----> IREmitter::visitNop: ");
  return nullptr;
}

Value* LLVMPass::IREmitter::visitMemoryGrow(wasm::MemoryGrow* curr) {
  LOG("  ----> IREmitter::visitMemoryGrow: ");
  auto name = Parent.memoryName(curr->memory.toString());
  auto& mem = Parent.namedMemories[name];
  int mem_size = mem.size;
  LOG(" .     Memory: " + name);

  Value* inst = nullptr;

  if (mem.max > mem.initial) {
    LOG(" .     Memory.Grow is not implemented");
  } else {
    Builder.CreateUnreachable();
  }
  inst = ConstantInt::get(Parent.i32, mem_size / wasm::Memory::kPageSize);

  return inst;
}