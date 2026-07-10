#include "shader_compilation.h"

#include <cstdint>
#include <cstring>

#include "log_macros.h"
#include "slang-com-ptr.h"
#include "slang.h"

namespace maple {
std::vector<uint8_t> compileSlangToSpirv(const std::string& code,
                                         const std::string& fileName,
                                         const std::string& vertEntryFuncName,
                                         const std::string& fragEntryFuncName) {
  Slang::ComPtr<slang::IGlobalSession> globalSession = nullptr;

  if (!globalSession) {
    if (SLANG_FAILED(slang::createGlobalSession(globalSession.writeRef()))) MAPLE_FATAL("failed to create slang global session");
  }

  slang::TargetDesc targetDesc = {};
  targetDesc.format = SLANG_SPIRV;
  targetDesc.profile = globalSession->findProfile("spirv_1_4");
  targetDesc.flags = 0;

  std::array<slang::CompilerOptionEntry, 3> options = {
    slang::CompilerOptionEntry{slang::CompilerOptionName::EmitSpirvDirectly, {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}},
    slang::CompilerOptionEntry{slang::CompilerOptionName::VulkanUseEntryPointName, {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}},
    slang::CompilerOptionEntry{slang::CompilerOptionName::MatrixLayoutColumn, {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}},
  };
  slang::SessionDesc sessionDesc = {};
  sessionDesc.targets = &targetDesc;
  sessionDesc.targetCount = 1;
  sessionDesc.preprocessorMacros = nullptr;
  sessionDesc.preprocessorMacroCount = 0;
  sessionDesc.compilerOptionEntries = options.data();
  sessionDesc.compilerOptionEntryCount = options.size();

  Slang::ComPtr<slang::ISession> session;
  globalSession->createSession(sessionDesc, session.writeRef());

  auto diagnoseIfNeeded = [](slang::IBlob* diagnosticsBlob) {
    if (diagnosticsBlob != nullptr) MAPLE_WARN((const char*)diagnosticsBlob->getBufferPointer());
  };

  Slang::ComPtr<slang::IModule> slangModule;
  {
    Slang::ComPtr<slang::IBlob> diagnosticsBlob;
    slangModule = session->loadModuleFromSourceString(fileName.c_str(), nullptr, code.c_str(), diagnosticsBlob.writeRef());
    diagnoseIfNeeded(diagnosticsBlob);
    if (!slangModule) MAPLE_FATAL("failed to load slang module");
  }

  Slang::ComPtr<slang::IEntryPoint> vertEntryPoint;
  {
    Slang::ComPtr<slang::IBlob> diagnosticsBlob;
    slangModule->findEntryPointByName(vertEntryFuncName.c_str(), vertEntryPoint.writeRef());
    diagnoseIfNeeded(diagnosticsBlob);
    if (!vertEntryPoint) MAPLE_FATAL("failed to find vertex entry point '{}'", vertEntryFuncName);
  }

  Slang::ComPtr<slang::IEntryPoint> fragEntryPoint;
  {
    Slang::ComPtr<slang::IBlob> diagnosticsBlob;
    slangModule->findEntryPointByName(fragEntryFuncName.c_str(), fragEntryPoint.writeRef());
    diagnoseIfNeeded(diagnosticsBlob);
    if (!fragEntryPoint) MAPLE_FATAL("failed to find fragment entry point '{}'", fragEntryFuncName);
  }

  std::array<slang::IComponentType*, 3> componentTypes = {
    slangModule,
    vertEntryPoint,
    fragEntryPoint,
  };

  Slang::ComPtr<slang::IComponentType> composedProgram;
  {
    Slang::ComPtr<slang::IBlob> diagnosticsBlob;
    SlangResult result =
      session->createCompositeComponentType(componentTypes.data(), componentTypes.size(), composedProgram.writeRef(), diagnosticsBlob.writeRef());
    diagnoseIfNeeded(diagnosticsBlob);
    if (SLANG_FAILED(result)) MAPLE_FATAL("");
  }

  Slang::ComPtr<slang::IComponentType> linkedProgram;
  {
    Slang::ComPtr<slang::IBlob> diagnosticsBlob;
    SlangResult result = composedProgram->link(linkedProgram.writeRef(), diagnosticsBlob.writeRef());
    diagnoseIfNeeded(diagnosticsBlob);
    if (SLANG_FAILED(result)) MAPLE_FATAL("");
  }

  Slang::ComPtr<slang::IBlob> spirvCode;
  {
    Slang::ComPtr<slang::IBlob> diagnosticsBlob;
    SlangResult result = linkedProgram->getTargetCode(0, spirvCode.writeRef(), diagnosticsBlob.writeRef());
    diagnoseIfNeeded(diagnosticsBlob);
    if (SLANG_FAILED(result)) MAPLE_FATAL("");
  }

  std::vector<uint8_t> result(spirvCode->getBufferSize());
  std::memcpy(result.data(), spirvCode->getBufferPointer(), spirvCode->getBufferSize());

  return result;
}
}  // namespace maple