# ClifftAmdgcn.cmake — compile plain-C device source to a .hsaco for the
# MLIR-V2 HIP-free path. Uses clang/llc/ld.lld from CLIFFT_LLVM_PREFIX (or PATH),
# targeting amdgcn-amd-amdhsa. NEVER uses -x hip or any HIP runtime.
#
# Proven viable in docs/v2/experiments (exp1/exp2): plain C with
# __builtin_amdgcn_* + extern addrspace(3) LDS -> valid HSA-loadable .hsaco.

# Locate the toolchain.
set(_clifft_llvm_hint "${CLIFFT_LLVM_PREFIX}")
if(NOT _clifft_llvm_hint AND DEFINED ENV{LLVM_PREFIX})
    set(_clifft_llvm_hint "$ENV{LLVM_PREFIX}")
endif()

find_program(CLIFFT_AMDGCN_CLANG  clang   HINTS "${_clifft_llvm_hint}/bin" /opt/rocm/lib/llvm/bin)
find_program(CLIFFT_AMDGCN_LLC    llc     HINTS "${_clifft_llvm_hint}/bin" /opt/rocm/lib/llvm/bin)
find_program(CLIFFT_AMDGCN_LLD    ld.lld  HINTS "${_clifft_llvm_hint}/bin" /opt/rocm/lib/llvm/bin)

if(NOT CLIFFT_AMDGCN_CLANG OR NOT CLIFFT_AMDGCN_LLC OR NOT CLIFFT_AMDGCN_LLD)
    message(FATAL_ERROR "ClifftAmdgcn: need clang, llc, ld.lld. Set CLIFFT_LLVM_PREFIX or LLVM_PREFIX. "
        "Found clang=${CLIFFT_AMDGCN_CLANG} llc=${CLIFFT_AMDGCN_LLC} lld=${CLIFFT_AMDGCN_LLD}")
endif()
message(STATUS "ClifftAmdgcn: clang=${CLIFFT_AMDGCN_CLANG} arch=${CLIFFT_AMDGPU_ARCH}")

# clifft_add_amdgcn_hsaco(<target> SOURCE <file.c> OUTPUT <name.hsaco> [EXTRA_FLAGS ...])
# Produces ${CMAKE_BINARY_DIR}/<name.hsaco> and a custom target <target>.
function(clifft_add_amdgcn_hsaco tgt)
    cmake_parse_arguments(A "" "SOURCE;OUTPUT" "EXTRA_FLAGS" ${ARGN})
    set(_src "${CMAKE_CURRENT_SOURCE_DIR}/${A_SOURCE}")
    set(_ll  "${CMAKE_CURRENT_BINARY_DIR}/${A_OUTPUT}.ll")
    set(_obj "${CMAKE_CURRENT_BINARY_DIR}/${A_OUTPUT}.o")
    set(_hsaco "${CMAKE_BINARY_DIR}/${A_OUTPUT}")
    add_custom_command(
        OUTPUT "${_hsaco}"
        # 1) plain C -> amdgcn LLVM-IR (freestanding, NO HIP, NO gpulib)
        COMMAND "${CLIFFT_AMDGCN_CLANG}" --target=amdgcn-amd-amdhsa -mcpu=${CLIFFT_AMDGPU_ARCH}
                -ffreestanding -nostdlib -nogpulib -std=c23 -O2 ${A_EXTRA_FLAGS}
                -emit-llvm -S -o "${_ll}" "${_src}"
        # 2) IR -> object
        COMMAND "${CLIFFT_AMDGCN_LLC}" -mtriple=amdgcn-amd-amdhsa -mcpu=${CLIFFT_AMDGPU_ARCH}
                -mattr=+wavefrontsize64 -filetype=obj -O2 -o "${_obj}" "${_ll}"
        # 3) object -> .hsaco (shared ELF HSA can load)
        COMMAND "${CLIFFT_AMDGCN_LLD}" -shared -o "${_hsaco}" "${_obj}"
        DEPENDS "${_src}"
        COMMENT "amdgcn (no-HIP): ${A_SOURCE} -> ${A_OUTPUT}"
        VERBATIM)
    add_custom_target(${tgt} ALL DEPENDS "${_hsaco}")
    set_property(GLOBAL APPEND PROPERTY CLIFFT_V2_HSACO "${_hsaco}")
endfunction()
