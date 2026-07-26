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

find_program(CLIFFT_AMDGCN_CLANG  clang     HINTS "${_clifft_llvm_hint}/bin" /opt/rocm/lib/llvm/bin)
find_program(CLIFFT_AMDGCN_LLC    llc       HINTS "${_clifft_llvm_hint}/bin" /opt/rocm/lib/llvm/bin)
find_program(CLIFFT_AMDGCN_LLD    ld.lld    HINTS "${_clifft_llvm_hint}/bin" /opt/rocm/lib/llvm/bin)
find_program(CLIFFT_AMDGCN_LINK   llvm-link HINTS "${_clifft_llvm_hint}/bin" /opt/rocm/lib/llvm/bin)
find_program(CLIFFT_AMDGCN_OPT    opt       HINTS "${_clifft_llvm_hint}/bin" /opt/rocm/lib/llvm/bin)

# ROCm device-library bitcode (ocml + oclc controls) for transcendentals
# (log/sincos). Located next to the ROCm clang. Isa-version bitcode is selected
# per-arch at compile time.
find_path(CLIFFT_ROCM_BITCODE_DIR ocml.bc
    HINTS /opt/rocm-7.2.3/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
          /opt/rocm/lib/llvm/lib/clang/*/lib/amdgcn/bitcode
          /opt/rocm/amdgcn/bitcode)

if(NOT CLIFFT_AMDGCN_CLANG OR NOT CLIFFT_AMDGCN_LLC OR NOT CLIFFT_AMDGCN_LLD)
    message(FATAL_ERROR "ClifftAmdgcn: need clang, llc, ld.lld. Set CLIFFT_LLVM_PREFIX or LLVM_PREFIX. "
        "Found clang=${CLIFFT_AMDGCN_CLANG} llc=${CLIFFT_AMDGCN_LLC} lld=${CLIFFT_AMDGCN_LLD}")
endif()
message(STATUS "ClifftAmdgcn: clang=${CLIFFT_AMDGCN_CLANG} arch=${CLIFFT_AMDGPU_ARCH}")

# clifft_add_amdgcn_hsaco(<target> SOURCE <file.c> OUTPUT <name.hsaco> [EXTRA_FLAGS ...])
# Produces ${CMAKE_BINARY_DIR}/<name.hsaco> and a custom target <target>.
# clifft_add_amdgcn_hsaco(<target> SOURCE <file.c> OUTPUT <name.hsaco>
#                         [EXTRA_FLAGS ...] [OCML])
# OCML: link ROCm ocml device library (for log/sincos/etc, byte-exact with SVM).
function(clifft_add_amdgcn_hsaco tgt)
    cmake_parse_arguments(A "OCML" "SOURCE;OUTPUT" "EXTRA_FLAGS" ${ARGN})
    set(_src "${CMAKE_CURRENT_SOURCE_DIR}/${A_SOURCE}")
    # The device headers are pulled in by every .c below but a custom command
    # gets no implicit header scanning -- depending on the .c file alone lets an
    # edit to v2_ops.h / v2_ops_body.inc leave a STALE .hsaco in place, silently
    # running the old kernel. (That is exactly how a barrier fix in v2_ops.h
    # nearly went unmeasured.) List them explicitly.
    # Glob is relative to the SOURCE's own directory so this stays correct
    # whichever subdirectory calls the function.
    get_filename_component(_src_dir "${_src}" DIRECTORY)
    file(GLOB _dev_hdrs "${_src_dir}/*.h" "${_src_dir}/*.inc")
    set(_ll  "${CMAKE_CURRENT_BINARY_DIR}/${A_OUTPUT}.ll")
    set(_obj "${CMAKE_CURRENT_BINARY_DIR}/${A_OUTPUT}.o")
    set(_hsaco "${CMAKE_BINARY_DIR}/${A_OUTPUT}")

    if(A_OCML)
        if(NOT CLIFFT_AMDGCN_LINK OR NOT CLIFFT_ROCM_BITCODE_DIR)
            message(FATAL_ERROR "ClifftAmdgcn OCML: need llvm-link and ROCm bitcode dir. "
                "llvm-link=${CLIFFT_AMDGCN_LINK} bitcode=${CLIFFT_ROCM_BITCODE_DIR}")
        endif()
        set(_bc "${CMAKE_CURRENT_BINARY_DIR}/${A_OUTPUT}.bc")
        set(_linked "${CMAKE_CURRENT_BINARY_DIR}/${A_OUTPUT}.linked.bc")
        # Map -mcpu -> oclc_isa_version_<num> control bitcode.
        string(REGEX REPLACE "^gfx" "" _isa "${CLIFFT_AMDGPU_ARCH}")
        set(_ctl "${CLIFFT_ROCM_BITCODE_DIR}")
        add_custom_command(
            OUTPUT "${_hsaco}"
            # 1) C -> amdgcn bitcode (freestanding, no HIP; ocml provides math)
            COMMAND "${CLIFFT_AMDGCN_CLANG}" --target=amdgcn-amd-amdhsa -mcpu=${CLIFFT_AMDGPU_ARCH}
                    -ffreestanding -nostdlib -nogpulib -std=c23 -O2 -ffp-contract=off ${A_EXTRA_FLAGS}
                    -emit-llvm -c -o "${_bc}" "${_src}"
            # 2) link ocml + oclc controls (wavefront64, daz/finite/unsafe off, isa)
            COMMAND "${CLIFFT_AMDGCN_LINK}" -o "${_linked}" "${_bc}"
                    "${_ctl}/ocml.bc" "${_ctl}/ockl.bc"
                    "${_ctl}/oclc_wavefrontsize64_on.bc"
                    "${_ctl}/oclc_daz_opt_off.bc"
                    "${_ctl}/oclc_finite_only_off.bc"
                    "${_ctl}/oclc_unsafe_math_off.bc"
                    "${_ctl}/oclc_correctly_rounded_sqrt_on.bc"
                    "${_ctl}/oclc_abi_version_500.bc"
                    "${_ctl}/oclc_isa_version_${_isa}.bc"
            # 3) internalize + optimize the linked module
            COMMAND "${CLIFFT_AMDGCN_OPT}" -O2 -o "${_linked}" "${_linked}"
            # 4) bitcode -> object
            COMMAND "${CLIFFT_AMDGCN_LLC}" -mtriple=amdgcn-amd-amdhsa -mcpu=${CLIFFT_AMDGPU_ARCH}
                    -mattr=+wavefrontsize64 -filetype=obj -O2 -o "${_obj}" "${_linked}"
            # 5) object -> .hsaco
            COMMAND "${CLIFFT_AMDGCN_LLD}" -shared -o "${_hsaco}" "${_obj}"
            DEPENDS "${_src}" ${_dev_hdrs}
            COMMENT "amdgcn (no-HIP, +ocml): ${A_SOURCE} -> ${A_OUTPUT}"
            VERBATIM)
    else()
        add_custom_command(
            OUTPUT "${_hsaco}"
            # 1) plain C -> amdgcn LLVM-IR (freestanding, NO HIP, NO gpulib)
            COMMAND "${CLIFFT_AMDGCN_CLANG}" --target=amdgcn-amd-amdhsa -mcpu=${CLIFFT_AMDGPU_ARCH}
                    -ffreestanding -nostdlib -nogpulib -std=c23 -O2 -ffp-contract=off ${A_EXTRA_FLAGS}
                    -emit-llvm -S -o "${_ll}" "${_src}"
            # 2) IR -> object
            COMMAND "${CLIFFT_AMDGCN_LLC}" -mtriple=amdgcn-amd-amdhsa -mcpu=${CLIFFT_AMDGPU_ARCH}
                    -mattr=+wavefrontsize64 -filetype=obj -O2 -o "${_obj}" "${_ll}"
            # 3) object -> .hsaco (shared ELF HSA can load)
            COMMAND "${CLIFFT_AMDGCN_LLD}" -shared -o "${_hsaco}" "${_obj}"
            DEPENDS "${_src}" ${_dev_hdrs}
            COMMENT "amdgcn (no-HIP): ${A_SOURCE} -> ${A_OUTPUT}"
            VERBATIM)
    endif()
    add_custom_target(${tgt} ALL DEPENDS "${_hsaco}")
    set_property(GLOBAL APPEND PROPERTY CLIFFT_V2_HSACO "${_hsaco}")
endfunction()
