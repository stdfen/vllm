// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "cpu/moe_mul1.h"
#include "parallel/all_reduce_cpu_avx2.h"
#include "parallel/all_reduce_cpu_avx512.h"

#include <stdexcept>
#include <utility>

namespace {
[[noreturn]] void unsupported() {
  throw std::runtime_error(
      "ExLlamaV3 x86 CPU backend is unavailable in the aarch64 CUDA-only build");
}
}  // namespace

bool is_avx2_supported() { return false; }
bool is_f16c_supported() { return false; }
bool is_avx512_supported() { return false; }
void enable_fast_fp() {}
void enable_fast_fp_avx2() {}
void enable_fast_fp_avx512() {}

void perform_cpu_reduce(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) {
  unsupported();
}
void perform_cpu_reduce_avx2(
    PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) {
  unsupported();
}
void perform_cpu_reduce_avx512(
    PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) {
  unsupported();
}
void cpu_reduce_parallel(
    void (*)(uint16_t*, const uint16_t*, const uint16_t*, size_t),
    void (*)(uint16_t*, const uint16_t*, size_t), uint16_t*, const uint16_t*,
    const uint16_t*, size_t, int) {
  unsupported();
}

void bf16_add_inplace_avx512(uint16_t*, const uint16_t*, size_t) {
  unsupported();
}
void bf16_add_twosrc_avx512(
    uint16_t*, const uint16_t*, const uint16_t*, size_t) {
  unsupported();
}
void fp16_add_inplace_avx512(uint16_t*, const uint16_t*, size_t) {
  unsupported();
}
void fp16_add_twosrc_avx512(
    uint16_t*, const uint16_t*, const uint16_t*, size_t) {
  unsupported();
}

int64_t exl3_moe_cpu_make_layer(
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    int64_t, double, int64_t) {
  unsupported();
}
void exl3_moe_cpu_free_layer(int64_t) {}
void exl3_moe_cpu_forward(
    int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    at::Tensor&, int64_t) {
  unsupported();
}
void exl3_moe_cpu_forward_raw(
    int64_t, const at::Half*, const int32_t*, const at::Half*, float*, int,
    int, int) {
  unsupported();
}
void exl3_moe_cpu_stage_experts(
    int64_t, const uint32_t*, int, uint8_t*, int) {
  unsupported();
}
void exl3_moe_cpu_set_prof(bool) {}
void exl3_moe_cpu_pool_prime(int) { unsupported(); }
int64_t exl3_moe_cpu_pool_stress(int, int, int, int) { unsupported(); }
std::pair<std::vector<int64_t>, int64_t> exl3_moe_cpu_core_order() {
  return {{}, 0};
}
bool exl3_moe_cpu_has_avx2() { return false; }
bool exl3_moe_cpu_has_avx512_bw() { return false; }
bool exl3_moe_cpu_has_avx512_vnni() { return false; }
bool exl3_moe_cpu_has_avx512_vbmi() { return false; }
