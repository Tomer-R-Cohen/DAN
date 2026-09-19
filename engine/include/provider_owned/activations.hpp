#pragma once

// Activation wire formats between stages (frame dtype):
//   f32  (1)  rows x cols float32
//   f16  (2)  rows x cols float16                      (half the bytes)
//   fp8  (3)  per row: float32 scale, cols x e4m3 bytes (a quarter of the bytes)
// Stages always compute in f32; only the bytes on the wire change.
//
// fp8 follows the design Shard (github.com/leyten/shard, v4_pipe.py) measured: e4m3 is a float
// with its own exponent, so a few very large hidden-state channels do not ruin the precision of
// the small ones (plain int8 does); and the scale is PER ROW (per token), so a token's bytes
// depend only on its own values, never on which other tokens share its frame. That keeps a
// speculative batch and a single-token step of the same position byte-identical.

#include "provider_owned/protocol.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>

namespace dan::provider_owned {

// Bytes of `rows` activations of width `cols` in this format (0 for non-activation dtypes).
inline std::uint64_t activation_bytes(DType dtype, std::uint64_t rows, std::uint64_t cols) {
    switch (dtype) {
    case DType::f32le: return rows * cols * 4;
    case DType::f16le: return rows * cols * 2;
    case DType::fp8e4m3: return rows * (cols + 4);
    default: return 0;
    }
}

inline const char* activation_name(DType dtype) {
    return dtype == DType::f16le ? "f16" : dtype == DType::fp8e4m3 ? "fp8" : "f32";
}

// e4m3 (the "fn" variant: no infinities, 0x7F/0xFF are NaN): 1 sign, 4 exponent (bias 7), 3
// mantissa bits; largest finite value 448, smallest subnormal 2^-9.
inline float fp8_e4m3_to_float(std::uint8_t code) {
    const int exponent = (code >> 3) & 0x0F;
    const int mantissa = code & 0x07;
    float value;
    if (exponent == 0x0F && mantissa == 0x07) value = NAN;
    else if (exponent == 0) value = std::ldexp(static_cast<float>(mantissa), -9);
    else value = std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
    return (code & 0x80) ? -value : value;
}

// Nearest e4m3 value (ties to even); saturates at +-448, NaN becomes 0.
inline std::uint8_t float_to_fp8_e4m3(float value) {
    if (std::isnan(value)) return 0;
    const std::uint8_t sign = std::signbit(value) ? 0x80 : 0;
    const float magnitude = std::fabs(value);
    if (magnitude >= 448.0f) return sign | 0x7E;
    if (magnitude == 0.0f) return sign;
    int exponent = 0;
    std::frexp(magnitude, &exponent);          // magnitude = m * 2^exponent, m in [0.5, 1)
    int biased = exponent - 1 + 7;             // for 1.xxx * 2^(exponent - 1)
    if (biased <= 0) {
        // Subnormal: multiples of 2^-9, up to the smallest normal (8 * 2^-9 = 2^-6).
        const int steps = static_cast<int>(std::nearbyint(std::ldexp(magnitude, 9)));
        return sign | static_cast<std::uint8_t>(steps);   // 8 lands exactly on code 0x08
    }
    int mantissa = static_cast<int>(std::nearbyint(
        (std::ldexp(magnitude, -(exponent - 1)) - 1.0f) * 8.0f));
    if (mantissa == 8) {
        mantissa = 0;
        ++biased;
    }
    if (biased > 15 || (biased == 15 && mantissa == 7)) return sign | 0x7E;
    return sign | static_cast<std::uint8_t>((biased << 3) | mantissa);
}

// One row to fp8: a float32 scale (big-endian bits) then `cols` e4m3 bytes. The scale is the
// row's largest finite magnitude / 448, never below 1e-8, so an all-zero row stays zero and one
// infinite value saturates alone instead of turning its whole row into NaN.
inline void pack_fp8_row(const float* source, std::size_t cols, std::uint8_t* target) {
    float largest = 0;
    for (std::size_t index = 0; index < cols; ++index) {
        const float magnitude = std::fabs(source[index]);
        if (std::isfinite(magnitude) && magnitude > largest) largest = magnitude;
    }
    const float scale = std::max(largest / 448.0f, 1e-8f);
    put32(target, std::bit_cast<std::uint32_t>(scale));
    for (std::size_t index = 0; index < cols; ++index) {
        target[4 + index] = float_to_fp8_e4m3(source[index] / scale);
    }
}

inline void unpack_fp8_row(const std::uint8_t* source, std::size_t cols, float* target) {
    static const std::array<float, 256> table = [] {
        std::array<float, 256> values{};
        for (int code = 0; code < 256; ++code) {
            values[static_cast<std::size_t>(code)] = fp8_e4m3_to_float(static_cast<std::uint8_t>(code));
        }
        return values;
    }();
    const float scale = std::bit_cast<float>(get32(source));
    for (std::size_t index = 0; index < cols; ++index) target[index] = table[source[4 + index]] * scale;
}

} // namespace dan::provider_owned
