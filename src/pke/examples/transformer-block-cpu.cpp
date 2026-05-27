#include "openfhe.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

using namespace lbcrypto;

/**
 * ============================================================================
 * 1. MATHEMATICAL AUXILIARY & APPROXIMATION PRIMITIVES
 * ============================================================================
 */

// Struct to hold Taylor expansion coefficients for f(z) = 1/sqrt(z) around a center z0.
// This is used as the initial seed generator for our Newton-Raphson inverse square root logic.
struct TaylorCoeffs {
    double c0, c1, c2, c3;
};

// Computes the Taylor expansion coefficients of 1/sqrt(z) around z0.
//   f(z)  = z^(-1/2)           => c0 = z0^(-1/2)
//   f'(z) = -1/2 * z^(-3/2)    => c1 = 1 / (2 * sqrt(z0^3))
//   f"(z) = 3/4 * z^(-5/2)     => c2 = 3 / (8 * sqrt(z0^5)) [incorporating the 1/2! factor]
//   f"'(z)= -15/8 * z^(-7/2)   => c3 = 5 / (16 * sqrt(z0^7)) [incorporating the 1/3! factor]
TaylorCoeffs computeTaylorCoeffs(double z0) {
    TaylorCoeffs t;
    t.c0 =  1.0 / std::sqrt(z0);
    t.c1 =  1.0 / (2.0 * std::sqrt(z0 * z0 * z0));
    t.c2 =  3.0 / (8.0 * std::sqrt(z0 * z0 * z0 * z0 * z0));
    t.c3 =  5.0 / (16.0 * std::sqrt(z0 * z0 * z0 * z0 * z0 * z0 * z0));
    return t;
}

// Helper utility to reduce the multiplicative level of a ciphertext down to a target level.
// This is required in CKKS to ensure that both operands of an addition or multiplication
// have identical scaling parameters and noise levels before evaluating the operation.
Ciphertext<DCRTPoly> levelReduceTo(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> ct,
    uint32_t targetLevel
) {
    uint32_t currentLevel = ct->GetLevel();
    if (currentLevel < targetLevel) {
        return cc->LevelReduce(ct, nullptr, targetLevel - currentLevel);
    }
    return ct;
}

// Evaluates the accelerated sign function using compositions of Chebyshev sign polynomials.
// We use the 9th-degree polynomials f4 (for the final iteration) and g4 (for intermediate iterations)
// as described in the literature (e.g., EncryptedLLM / FHE-transformer papers).
//   df: number of compositions with f4
//   dg: number of compositions with g4
Ciphertext<DCRTPoly> Eval_accelerated_sign(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> x,
    int df,
    int dg
) {
    // Coefficients derived from Chebyshev approximation of sign(x) on range [-1, 1]
    std::vector<double> coeff_f4 = {0.0, 315.0/128.0, 0.0, -420.0/128.0, 0.0, 378.0/128.0, 0.0, -180.0/128.0, 0.0, 35.0/128.0};
    std::vector<double> coeff_g4 = {0.0, 5850.0/1024.0, 0.0, -34974.0/1024.0, 0.0, 97015.0/1024.0, 0.0, -113492.0/1024.0, 0.0, 46623.0/1024.0};

    auto res = x;
    // Step 1: Pre-convergence composition using g4
    for (int i = 0; i < dg; ++i) {
        res = cc->EvalPoly(res, coeff_g4);
    }
    // Step 2: Final shaping composition using f4
    for (int i = 0; i < df; ++i) {
        res = cc->EvalPoly(res, coeff_f4);
    }
    return res;
}

// Element-wise maximum of two ciphertexts: max(a, b).
// We approximate this using the identity: max(a, b) = 0.5 * (a + b + |a - b|).
// The absolute value is computed by: |x| = x * sign(x).
// To prevent the Chebyshev polynomial from diverging, we scale the difference to fit in [-1, 1].
Ciphertext<DCRTPoly> approxMax2(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> a,
    Ciphertext<DCRTPoly> b,
    double scale,
    int df,
    int dg
) {
    // Align levels of inputs to the deepest level among them
    uint32_t max_input_lvl = std::max(a->GetLevel(), b->GetLevel());
    auto a_aligned = levelReduceTo(cc, a, max_input_lvl);
    auto b_aligned = levelReduceTo(cc, b, max_input_lvl);

    // Compute diff = a - b
    auto diff = cc->EvalSub(a_aligned, b_aligned);
    
    // Scale diff to prevent sign polynomial divergence outside [-1, 1]
    auto diff_scaled = cc->EvalMult(diff, 1.0 / scale);
    
    // Evaluate sign(diff_scaled)
    auto sign_val = Eval_accelerated_sign(cc, diff_scaled, df, dg);

    // Compute |diff| = diff * sign(diff)
    uint32_t sign_lvl = sign_val->GetLevel();
    auto diff_leveled = levelReduceTo(cc, diff, sign_lvl);
    auto abs_diff = cc->EvalMult(diff_leveled, sign_val);

    // Compute sum = a + b
    auto sum = cc->EvalAdd(a_aligned, b_aligned);

    // final max calculation: 0.5 * (sum + |diff|)
    uint32_t abs_diff_lvl = abs_diff->GetLevel();
    auto sum_leveled = levelReduceTo(cc, sum, abs_diff_lvl);

    auto sum_abs = cc->EvalAdd(sum_leveled, abs_diff);
    return cc->EvalMult(sum_abs, 0.5);
}

// Replicate slot 0 values to the first 'count' slots using a binary rotation tree.
// Useful for broadcasting scalar values (like row max or row sum) across an entire vector.
Ciphertext<DCRTPoly> ReplicateSlot0(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> ct,
    uint32_t count
) {
    // Mask out all slots except slot 0
    std::vector<double> mask0(ct->GetSlots(), 0.0);
    mask0[0] = 1.0;
    Plaintext ptMask0 = cc->MakeCKKSPackedPlaintext(mask0, 1, 0, nullptr, ct->GetSlots());
    Ciphertext<DCRTPoly> result = cc->EvalMult(ct, ptMask0);

    // Accumulate sum via log2(count) rotations
    uint32_t currentReplicas = 1;
    while (currentReplicas < count) {
        uint32_t shift = currentReplicas;
        auto rotated = cc->EvalRotate(result, -static_cast<int32_t>(shift));
        result = cc->EvalAdd(result, rotated);
        currentReplicas *= 2;
    }
    return result;
}

// Tree-based maximum reduction over the first 'n' slots.
// Performs log2(n) iterations of approxMax2, folding the vector in half each time,
// and then replicates the final maximum back across all 'n' slots.
Ciphertext<DCRTPoly> fheMax(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> x,
    uint32_t n,
    double scale,
    int df,
    int dg
) {
    auto current_max = x;
    for (uint32_t step = n / 2; step > 0; step /= 2) {
        auto rotated = cc->EvalRotate(current_max, step);
        current_max = approxMax2(cc, current_max, rotated, scale, df, dg);
    }
    return ReplicateSlot0(cc, current_max, n);
}

// Approximates exp(x) using the identity: exp(x) = (1 + x / 2^r)^(2^r).
// This is evaluated by performing r squarings, requiring exactly r levels.
Ciphertext<DCRTPoly> approxExp(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> x,
    int r
) {
    double divisor = static_cast<double>(1 << r);
    // Base = 1.0 + x / 2^r
    auto base = cc->EvalAdd(x, divisor);
    base = cc->EvalMult(base, 1.0 / divisor);

    // Squaring loop: base^(2^r)
    auto res = base;
    for (int i = 0; i < r; ++i) {
        res = cc->EvalMult(res, res);
    }
    return res;
}

// Goldschmidt Division for homomorphic division N / D.
// Scale both numerator and denominator such that D_0 = D * F_0 is close to 1.
// Then run Newton-Raphson division iterations:
//   F_i   = 2 - D_i
//   N_i+1 = N_i * F_i
//   D_i+1 = D_i * F_i
Ciphertext<DCRTPoly> goldschmidtDivision(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> N,
    Ciphertext<DCRTPoly> D,
    uint32_t seq_len,
    int iterations
) {
    uint32_t max_lvl = std::max(N->GetLevel(), D->GetLevel());
    auto N_curr = levelReduceTo(cc, N, max_lvl);
    auto D_curr = levelReduceTo(cc, D, max_lvl);

    // Scaling factor: map expected sum-exp range (around active length) to the convergence interval
    double F0 = 1.0 / (static_cast<double>(seq_len) * 1.5);
    N_curr = cc->EvalMult(N_curr, F0);
    D_curr = cc->EvalMult(D_curr, F0);

    for (int i = 0; i < iterations; ++i) {
        auto F = cc->EvalSub(2.0, D_curr);
        N_curr = cc->EvalMult(N_curr, F);
        D_curr = cc->EvalMult(D_curr, F);
    }
    return N_curr;
}

// Row-wise active Softmax implementation.
// Performs numerical stabilization by subtracting the row maximum (fheMax) to avoid out-of-bounds
// divergence of the polynomial exponentiation, then computes exp(x - max) and divides by the sum of exponents.
Ciphertext<DCRTPoly> approxSoftmax(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> x,
    uint32_t n, // active elements in the row (e.g. i+1 for causal masking)
    double scale,
    int df,
    int dg,
    int r,
    int goldschmidt_iters
) {
    // Find maximum of the first 'n' slots
    auto x_max = fheMax(cc, x, n, scale, df, dg);
    uint32_t max_lvl = x_max->GetLevel();
    auto x_leveled = levelReduceTo(cc, x, max_lvl);
    
    // Shift input to negative domain: x - max
    auto x_shifted = cc->EvalSub(x_leveled, x_max);

    // Compute exponents: exp(x - max)
    auto exp_x = approxExp(cc, x_shifted, r);
    
    // Sum of exponents
    auto sum_exp = cc->EvalSum(exp_x, n);
    sum_exp = ReplicateSlot0(cc, sum_exp, n); // Broadcast sum to all n slots
    
    // Divide element-wise: exp_x / sum_exp
    auto out = goldschmidtDivision(cc, exp_x, sum_exp, n, goldschmidt_iters);
    return out;
}

/**
 * ============================================================================
 * 2. LAYER NORMALIZATION (ROW-WISE STABILIZED)
 * ============================================================================
 */

// Approximates LayerNorm on a single vector using a Taylor-Newton pipeline:
//   1. Mean: mu = Sum(x) / n
//   2. Variance: sigma^2 = Sum((x - mu)^2) / n
//   3. Inverse standard deviation: inv_std = 1 / sqrt(sigma^2 + eps)
//      Pre-scaled to 1.0 around z0, expanded with Taylor series, and refined using Newton-Raphson.
//   4. Output: (x - mu) * inv_std * weight + bias
Ciphertext<DCRTPoly> approxLayerNorm(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> x,
    uint32_t n,
    const std::vector<double>& weight,
    const std::vector<double>& bias,
    double eps,
    double z0,
    int newton_iters
) {
    // Pre-scale variables to bring expected variance close to 1.0 for Taylor convergence
    double k = 1.0 / std::sqrt(z0);
    double eps_scaled = eps / z0;
    double z0_scaled = 1.0;

    auto x_scaled = cc->EvalMult(x, k);
    auto sum_x = cc->EvalSum(x_scaled, n);
    sum_x = ReplicateSlot0(cc, sum_x, n); // Broadcast sum to all n slots
    auto mu = cc->EvalMult(sum_x, 1.0 / static_cast<double>(n));

    // Centered values: x - mu
    auto x_centered = cc->EvalSub(x_scaled, mu);
    auto x_centered_sq = cc->EvalMult(x_centered, x_centered);
    auto sum_sq = cc->EvalSum(x_centered_sq, n);
    sum_sq = ReplicateSlot0(cc, sum_sq, n); // Broadcast sum to all n slots
    auto sigma_sq = cc->EvalMult(sum_sq, 1.0 / static_cast<double>(n));
    
    // Variance + epsilon
    auto z = cc->EvalAdd(sigma_sq, eps_scaled);

    // Taylor expansion of 1/sqrt(z) around z0_scaled = 1.0
    TaylorCoeffs tc = computeTaylorCoeffs(z0_scaled);
    auto z_diff = cc->EvalSub(z, z0_scaled);
    auto z_diff_sq = cc->EvalMult(z_diff, z_diff);
    auto z_diff_cub = cc->EvalMult(z_diff_sq, z_diff);

    // y = c0 - c1*z_diff + c2*z_diff^2 - c3*z_diff^3
    auto y = cc->EvalMult(z_diff, tc.c1);
    y = cc->EvalSub(tc.c0, y);
    auto term2 = cc->EvalMult(z_diff_sq, tc.c2);
    y = cc->EvalAdd(y, term2);
    auto term3 = cc->EvalMult(z_diff_cub, tc.c3);
    y = cc->EvalSub(y, term3);

    // Newton iterations: y = y * (3.0 - z * y^2) / 2.0
    for (int i = 0; i < newton_iters; ++i) {
        auto y_sq = cc->EvalMult(y, y);
        auto z_y_sq = cc->EvalMult(z, y_sq);
        auto inner = cc->EvalSub(3.0, z_y_sq);
        auto y_inner = cc->EvalMult(y, inner);
        y = cc->EvalMult(y_inner, 0.5);
    }

    // Multiply centered values by inverse standard deviation: (x - mu) * y
    auto x_norm = cc->EvalMult(x_centered, y);
    
    // Resize weight and bias to match the ciphertext slot layout exactly (avoids slot corruption)
    uint32_t slots = x->GetSlots();
    std::vector<double> weight_resized(slots, 0.0);
    std::vector<double> bias_resized(slots, 0.0);
    for (uint32_t i = 0; i < weight.size() && i < slots; ++i) weight_resized[i] = weight[i];
    for (uint32_t i = 0; i < bias.size() && i < slots; ++i) bias_resized[i] = bias[i];
    Plaintext ptWeight = cc->MakeCKKSPackedPlaintext(weight_resized, 1, 0, nullptr, slots);
    Plaintext ptBias = cc->MakeCKKSPackedPlaintext(bias_resized, 1, 0, nullptr, slots);

    // Apply affine parameters: weight * x_norm + bias
    auto result = cc->EvalMult(x_norm, ptWeight);
    result = cc->EvalAdd(result, ptBias);
    return result;
}

// Row-wise LayerNorm wrapper for L x D matrix layout.
// Evaluates approxLayerNorm independently on each row of the matrix.
// This is achieved by extracting each row, rotating it to slot 0, performing the LN, and masking/shifting it back.
Ciphertext<DCRTPoly> EvalRowWiseLayerNorm(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> ctX,
    uint32_t L, uint32_t D,
    const std::vector<double>& weight,
    const std::vector<double>& bias,
    double eps, double z0, int newton_iters
) {
    uint32_t slots = std::max<uint32_t>(ctX->GetSlots(), L * D);
    auto ctX_working = ctX->Clone();
    ctX_working->SetSlots(slots);
    std::vector<Ciphertext<DCRTPoly>> rowResults(L);

    // Mask covering only the first D slots (the active row width)
    std::vector<double> maskD(slots, 0.0);
    std::fill(maskD.begin(), maskD.begin() + D, 1.0);
    Plaintext ptMaskD = cc->MakeCKKSPackedPlaintext(maskD, 1, 0, nullptr, slots);

    for (uint32_t i = 0; i < L; ++i) {
        // Step 1: Extract row i to slot [0..D-1] via rotation and masking
        Ciphertext<DCRTPoly> ctRow = (i * D == 0) ? ctX_working : cc->EvalRotate(ctX_working, i * D);
        ctRow = cc->EvalMult(ctRow, ptMaskD);

        // Step 2: Evaluate LayerNorm on the isolated row
        auto ctRowLN = approxLayerNorm(cc, ctRow, D, weight, bias, eps, z0, newton_iters);

        // Step 3: Shift the normalized row back to its original row slot position
        Ciphertext<DCRTPoly> ctRowBack = (i * D == 0) ? ctRowLN : cc->EvalRotate(ctRowLN, -static_cast<int32_t>(i * D));
        std::vector<double> maskRowi(slots, 0.0);
        std::fill(maskRowi.begin() + i * D, maskRowi.begin() + (i + 1) * D, 1.0);
        Plaintext ptMaskRowi = cc->MakeCKKSPackedPlaintext(maskRowi, 1, 0, nullptr, slots);
        ctRowBack = cc->EvalMult(ctRowBack, ptMaskRowi);

        rowResults[i] = ctRowBack;
    }

    // Step 4: Align the levels of all computed row ciphertexts and accumulate them
    uint32_t maxLvl = 0;
    for (uint32_t i = 0; i < L; ++i) {
        maxLvl = std::max<uint32_t>(maxLvl, rowResults[i]->GetLevel());
    }
    Ciphertext<DCRTPoly> result;
    for (uint32_t i = 0; i < L; ++i) {
        rowResults[i] = levelReduceTo(cc, rowResults[i], maxLvl);
        if (i == 0) {
            result = rowResults[i];
        } else {
            result = cc->EvalAdd(result, rowResults[i]);
        }
    }
    result->SetSlots(slots);
    return result;
}

/**
 * ============================================================================
 * 3. PIECEWISE GELU ACTIVATION
 * ============================================================================
 */

// Compares if input x is greater than a threshold using: 0.5 * (sign(x - threshold) + 1.0).
Ciphertext<DCRTPoly> Eval_accelerated_comparison(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> x,
    double threshold,
    int df,
    int dg
) {
    auto diff = cc->EvalSub(x, threshold);
    // Scale by 1/16 to ensure the difference falls within the stable sign-convergence region
    diff = cc->EvalMult(diff, 1.0 / 16.0);
    auto sign_val = Eval_accelerated_sign(cc, diff, df, dg);
    auto res = cc->EvalAdd(sign_val, 1.0);
    return cc->EvalMult(res, 0.5);
}

// Piecewise element-wise approximation of the GELU activation function.
// Splits the input domain into 4 regions to balance polynomial degree and error:
//   - Branch 0: x <= -4.0      => 0.0
//   - Branch 1: -4.0 < x <= -1.95 => 3rd-degree polynomial f0(x)
//   - Branch 2: -1.95 < x <= 3.0 => 6th-degree polynomial f1(x)
//   - Branch 3: x > 3.0        => x
Ciphertext<DCRTPoly> approx_gelu_piecewise(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> x,
    int df,
    int dg
) {
    // Step 1: Generate indicator variables for the bounds
    auto c_neg4 = Eval_accelerated_comparison(cc, x, -4.0, df, dg);
    auto c_neg1_95 = Eval_accelerated_comparison(cc, x, -1.95, df, dg);
    auto c_3 = Eval_accelerated_comparison(cc, x, 3.0, df, dg);

    // Step 2: Build branch selection masks (multiplexing)
    auto ind_0 = cc->EvalSub(1.0, c_neg4);
    auto ind_1_sub = cc->EvalSub(1.0, c_neg1_95);
    auto ind_1 = cc->EvalMult(c_neg4, ind_1_sub);
    auto ind_2_sub = cc->EvalSub(1.0, c_3);
    auto ind_2 = cc->EvalMult(c_neg1_95, ind_2_sub);
    auto ind_3 = c_3;

    // Piecewise polynomial coefficients
    std::vector<double> coeff_f0 = {-0.5054031199708174, -0.42226581151983866, -0.11807612951181953, -0.011034134030615728};
    std::vector<double> coeff_f1 = {0.008526321541038084, 0.5, 0.3603292692789629, 0.0, -0.037688200365904236, 0.0, 0.0018067462606141187};

    // Step 3: Evaluate function values on all branches
    auto val_1 = cc->EvalPoly(x, coeff_f0);
    auto val_2 = cc->EvalPoly(x, coeff_f1);
    auto val_3 = x;

    // Step 4: Multiply branch indicator masks with branch values and sum them
    uint32_t targetLvl = std::max({ind_1->GetLevel(), val_1->GetLevel(), ind_2->GetLevel(), val_2->GetLevel(), ind_3->GetLevel(), val_3->GetLevel()});
    auto ind_1_aligned = levelReduceTo(cc, ind_1, targetLvl);
    auto val_1_aligned = levelReduceTo(cc, val_1, targetLvl);
    auto ind_2_aligned = levelReduceTo(cc, ind_2, targetLvl);
    auto val_2_aligned = levelReduceTo(cc, val_2, targetLvl);
    auto ind_3_aligned = levelReduceTo(cc, ind_3, targetLvl);
    auto val_3_aligned = levelReduceTo(cc, val_3, targetLvl);

    auto branch1 = cc->EvalMult(ind_1_aligned, val_1_aligned);
    auto branch2 = cc->EvalMult(ind_2_aligned, val_2_aligned);
    auto branch3 = cc->EvalMult(ind_3_aligned, val_3_aligned);

    auto result = cc->EvalAdd(branch1, branch2);
    result = cc->EvalAdd(result, branch3);
    return result;
}

/**
 * ============================================================================
 * 4. FHE MATRIX MULTIPLICATION PRIMITIVES
 * ============================================================================
 */

// Rectangular matrix multiplication: Ciphertext ctX (L x Din) * Plaintext Matrix W (Din x Dout).
// Computes row-wise matrix product: isolates each row of the input, multiplies it element-wise
// with column vectors of W, sums them up, and shifts them to build the output matrix ciphertext.
Ciphertext<DCRTPoly> EvalMatMulCtPtRect(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> ctX,
    const std::vector<std::vector<double>>& W,
    uint32_t L, uint32_t Din, uint32_t Dout
) {
    uint32_t slots = std::max<uint32_t>(ctX->GetSlots(), L * std::max(Din, Dout));
    auto ctX_working = ctX->Clone();
    ctX_working->SetSlots(slots);
    std::vector<Ciphertext<DCRTPoly>> terms;

    for (uint32_t i = 0; i < L; ++i) {
        // Isolate row i of ctX
        Ciphertext<DCRTPoly> ctRow = (i * Din == 0) ? ctX_working : cc->EvalRotate(ctX_working, i * Din);
        std::vector<double> maskDin(slots, 0.0);
        std::fill(maskDin.begin(), maskDin.begin() + Din, 1.0);
        Plaintext ptMaskDin = cc->MakeCKKSPackedPlaintext(maskDin, 1, 0, nullptr, slots);
        ctRow = cc->EvalMult(ctRow, ptMaskDin);

        for (uint32_t j = 0; j < Dout; ++j) {
            // Isolate column j of weight matrix W
            std::vector<double> colW(slots, 0.0);
            for (uint32_t k = 0; k < Din; ++k) {
                colW[k] = W[k][j];
            }
            Plaintext ptColW = cc->MakeCKKSPackedPlaintext(colW, 1, 0, nullptr, slots);
            
            // Dot product between ctRow and colW
            Ciphertext<DCRTPoly> prod = cc->EvalMult(ctRow, ptColW);
            Ciphertext<DCRTPoly> sum = cc->EvalSum(prod, Din);

            // Shift product result to target slot targetSlot = i * Dout + j
            uint32_t targetSlot = i * Dout + j;
            Ciphertext<DCRTPoly> shifted = (targetSlot == 0) ? sum : cc->EvalRotate(sum, -static_cast<int32_t>(targetSlot));

            // Mask target slot
            std::vector<double> targetMask(slots, 0.0);
            targetMask[targetSlot] = 1.0;
            Plaintext ptTargetMask = cc->MakeCKKSPackedPlaintext(targetMask, 1, 0, nullptr, slots);
            Ciphertext<DCRTPoly> masked = cc->EvalMult(shifted, ptTargetMask);
            terms.push_back(masked);
        }
    }

    // Align levels and sum all terms
    uint32_t maxLvl = 0;
    for (const auto& term : terms) {
        maxLvl = std::max<uint32_t>(maxLvl, term->GetLevel());
    }

    Ciphertext<DCRTPoly> result = levelReduceTo(cc, terms[0], maxLvl);
    for (size_t idx = 1; idx < terms.size(); ++idx) {
        result = cc->EvalAdd(result, levelReduceTo(cc, terms[idx], maxLvl));
    }
    result->SetSlots(slots);
    return result;
}

// Ciphertext-Ciphertext matrix product for Q (L x D) * K^T (D x L) -> Scores (L x L).
// Since both matrices are encrypted, we compute dot products between row i of Q and row j of K.
Ciphertext<DCRTPoly> EvalMatMulCtCtQTKT(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> ctQ,
    Ciphertext<DCRTPoly> ctK,
    uint32_t L, uint32_t D
) {
    uint32_t slots = std::max<uint32_t>(ctQ->GetSlots(), std::max(L * D, L * L));
    auto ctQ_working = ctQ->Clone(); ctQ_working->SetSlots(slots);
    auto ctK_working = ctK->Clone(); ctK_working->SetSlots(slots);
    std::vector<double> maskD(slots, 0.0);
    std::fill(maskD.begin(), maskD.begin() + D, 1.0);
    Plaintext ptMaskD = cc->MakeCKKSPackedPlaintext(maskD, 1, 0, nullptr, slots);

    std::vector<Ciphertext<DCRTPoly>> dotProducts(L * L);

    for (uint32_t i = 0; i < L; ++i) {
        // Extract row i of Q
        Ciphertext<DCRTPoly> ctQ_i = (i * D == 0) ? ctQ_working : cc->EvalRotate(ctQ_working, i * D);
        ctQ_i = cc->EvalMult(ctQ_i, ptMaskD);

        for (uint32_t j = 0; j < L; ++j) {
            // Extract row j of K (representing column j of K^T)
            Ciphertext<DCRTPoly> ctK_j = (j * D == 0) ? ctK_working : cc->EvalRotate(ctK_working, j * D);
            ctK_j = cc->EvalMult(ctK_j, ptMaskD);

            // Compute dot product: Q_i * K_j
            Ciphertext<DCRTPoly> prod = cc->EvalMult(ctQ_i, ctK_j);
            Ciphertext<DCRTPoly> sum = cc->EvalSum(prod, D);

            dotProducts[i * L + j] = sum;
        }
    }

    // Merge the individual dot product ciphertexts into a single layout
    return cc->EvalMerge(dotProducts);
}

// Ciphertext-Ciphertext matrix product for Scores (L x L) * V (L x D) -> Attention Output (L x D).
// Multiplying intermediate attention coefficients (Scores) with values (V).
Ciphertext<DCRTPoly> EvalMatMulCtCtScoresV(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> ctScores,
    Ciphertext<DCRTPoly> ctV,
    uint32_t L, uint32_t D
) {
    uint32_t slots = std::max<uint32_t>(ctScores->GetSlots(), std::max(L * D, L * L));
    auto ctScores_working = ctScores->Clone(); ctScores_working->SetSlots(slots);
    auto ctV_working = ctV->Clone(); ctV_working->SetSlots(slots);
    std::vector<double> maskD(slots, 0.0);
    std::fill(maskD.begin(), maskD.begin() + D, 1.0);
    Plaintext ptMaskD = cc->MakeCKKSPackedPlaintext(maskD, 1, 0, nullptr, slots);

    std::vector<Ciphertext<DCRTPoly>> rowResults(L);

    for (uint32_t i = 0; i < L; ++i) {
        Ciphertext<DCRTPoly> ctRowSum;
        bool isFirst = true;

        for (uint32_t k = 0; k < L; ++k) {
            // Isolate scalar weight scores[i, k]
            Ciphertext<DCRTPoly> ctScalar = (i * L + k == 0) ? ctScores_working : cc->EvalRotate(ctScores_working, i * L + k);
            Ciphertext<DCRTPoly> ctScalarRep = ReplicateSlot0(cc, ctScalar, D);

            // Extract row k of values matrix V
            Ciphertext<DCRTPoly> ctV_k = (k * D == 0) ? ctV_working : cc->EvalRotate(ctV_working, k * D);
            ctV_k = cc->EvalMult(ctV_k, ptMaskD);

            // Scale alignment
            uint32_t scalarLvl = ctScalarRep->GetLevel();
            ctV_k = levelReduceTo(cc, ctV_k, scalarLvl);

            // Multiply row V_k by scalar score
            Ciphertext<DCRTPoly> term = cc->EvalMult(ctScalarRep, ctV_k);

            if (isFirst) {
                ctRowSum = term;
                isFirst = false;
            } else {
                ctRowSum = cc->EvalAdd(ctRowSum, term);
            }
        }

        rowResults[i] = ctRowSum;
    }

    // Assemble the rows back to build the full L x D matrix layout
    Ciphertext<DCRTPoly> result;
    bool isFirst = true;
    for (uint32_t i = 0; i < L; ++i) {
        std::vector<double> maskRow(slots, 0.0);
        std::fill(maskRow.begin() + i * D, maskRow.begin() + (i + 1) * D, 1.0);
        Plaintext ptMaskRow = cc->MakeCKKSPackedPlaintext(maskRow, 1, 0, nullptr, slots);

        Ciphertext<DCRTPoly> shiftedRow = (i * D == 0) ? rowResults[i] : cc->EvalRotate(rowResults[i], -static_cast<int32_t>(i * D));
        Ciphertext<DCRTPoly> maskedRow = cc->EvalMult(shiftedRow, ptMaskRow);

        if (isFirst) {
            result = maskedRow;
            isFirst = false;
        } else {
            result = cc->EvalAdd(result, maskedRow);
        }
    }
    result->SetSlots(slots);
    return result;
}

/**
 * ============================================================================
 * 5. CAUSAL SELF-ATTENTION VERIFIER
 * ============================================================================
 */

// Causal Self-Attention Block.
// Executes standard attention: projections Q, K, V -> QK^T -> causal masking + Softmax -> Scores x V -> Out-projection.
// Can decrypt and print intermediate stages for detailed trace debugging if secretKey is provided.
Ciphertext<DCRTPoly> EvalCausalAttention(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> ctX,
    uint32_t L, uint32_t D,
    const std::vector<std::vector<double>>& W_Q,
    const std::vector<std::vector<double>>& W_K,
    const std::vector<std::vector<double>>& W_V,
    const std::vector<std::vector<double>>& W_O,
    double softmaxScale, int df, int dg, int r, int goldschmidt_iters,
    PrivateKey<DCRTPoly> secretKey = nullptr
) {
    uint32_t slots = std::max<uint32_t>(ctX->GetSlots(), std::max(L * D, L * L));
    auto ctX_working = ctX->Clone();
    ctX_working->SetSlots(slots);

    // 1. Projections Q, K, V
    std::cout << "  Computing projections Q, K, V..." << std::endl;
    auto ctQ = EvalMatMulCtPtRect(cc, ctX_working, W_Q, L, D, D);
    auto ctK = EvalMatMulCtPtRect(cc, ctX_working, W_K, L, D, D);
    auto ctV = EvalMatMulCtPtRect(cc, ctX_working, W_V, L, D, D);

    // Intermediate tracing for projected matrices
    if (secretKey) {
        Plaintext pt;
        cc->Decrypt(secretKey, ctQ, &pt); pt->SetLength(L * D);
        auto val = pt->GetCKKSPackedValue();
        std::cout << "    [DEBUG ctQ] level: " << ctQ->GetLevel() << ", values: [ ";
        for (auto& c : val) { std::cout << c.real() << " "; } std::cout << "]" << std::endl;

        cc->Decrypt(secretKey, ctK, &pt); pt->SetLength(L * D);
        val = pt->GetCKKSPackedValue();
        std::cout << "    [DEBUG ctK] level: " << ctK->GetLevel() << ", values: [ ";
        for (auto& c : val) { std::cout << c.real() << " "; } std::cout << "]" << std::endl;

        cc->Decrypt(secretKey, ctV, &pt); pt->SetLength(L * D);
        val = pt->GetCKKSPackedValue();
        std::cout << "    [DEBUG ctV] level: " << ctV->GetLevel() << ", values: [ ";
        for (auto& c : val) { std::cout << c.real() << " "; } std::cout << "]" << std::endl;
    }

    // 2. Q x K^T
    std::cout << "  Evaluating Q x K^T..." << std::endl;
    auto ctScores = EvalMatMulCtCtQTKT(cc, ctQ, ctK, L, D);

    // Intermediate tracing for attention scores
    if (secretKey) {
        Plaintext pt;
        cc->Decrypt(secretKey, ctScores, &pt); pt->SetLength(L * L);
        auto val = pt->GetCKKSPackedValue();
        std::cout << "    [DEBUG ctScores (QK^T)] level: " << ctScores->GetLevel() << ", values: [ ";
        for (auto& c : val) { std::cout << c.real() << " "; } std::cout << "]" << std::endl;
    }

    // 3. Row-wise active Softmax with causal masking
    std::cout << "  Evaluating row-wise softmax..." << std::endl;
    Ciphertext<DCRTPoly> ctCoeffs;
    std::vector<Ciphertext<DCRTPoly>> rowResults(L);

    for (uint32_t i = 0; i < L; ++i) {
        // Extract row i of scores
        Ciphertext<DCRTPoly> ctRow = (i * L == 0) ? ctScores : cc->EvalRotate(ctScores, i * L);
        
        // Causal Masking: zero out inactive future elements (slots index > i)
        std::vector<double> maskActive(slots, 0.0);
        std::fill(maskActive.begin(), maskActive.begin() + i + 1, 1.0);
        Plaintext ptMaskActive = cc->MakeCKKSPackedPlaintext(maskActive, 1, 0, nullptr, slots);
        ctRow = cc->EvalMult(ctRow, ptMaskActive);

        // Evaluate Softmax only over the active row length (i + 1 elements) to prevent sign poly divergence
        Ciphertext<DCRTPoly> ctRowSoftmax = approxSoftmax(cc, ctRow, i + 1, softmaxScale, df, dg, r, goldschmidt_iters);

        // Shift row back to its matrix layout position
        Ciphertext<DCRTPoly> ctRowBack = (i * L == 0) ? ctRowSoftmax : cc->EvalRotate(ctRowSoftmax, -static_cast<int32_t>(i * L));
        std::vector<double> maskRowi(slots, 0.0);
        std::fill(maskRowi.begin() + i * L, maskRowi.begin() + i * L + i + 1, 1.0);
        Plaintext ptMaskRowi = cc->MakeCKKSPackedPlaintext(maskRowi, 1, 0, nullptr, slots);
        ctRowBack = cc->EvalMult(ctRowBack, ptMaskRowi);

        rowResults[i] = ctRowBack;
    }

    // Align levels and accumulate attention coefficients
    uint32_t maxLvl = 0;
    for (uint32_t i = 0; i < L; ++i) {
        maxLvl = std::max<uint32_t>(maxLvl, rowResults[i]->GetLevel());
    }
    for (uint32_t i = 0; i < L; ++i) {
        rowResults[i] = levelReduceTo(cc, rowResults[i], maxLvl);
        if (i == 0) {
            ctCoeffs = rowResults[i];
        } else {
            ctCoeffs = cc->EvalAdd(ctCoeffs, rowResults[i]);
        }
    }

    // Intermediate tracing for attention coefficients
    if (secretKey) {
        Plaintext pt;
        cc->Decrypt(secretKey, ctCoeffs, &pt); pt->SetLength(L * L);
        auto val = pt->GetCKKSPackedValue();
        std::cout << "    [DEBUG ctCoeffs (Softmax)] level: " << ctCoeffs->GetLevel() << ", values: [ ";
        for (auto& c : val) { std::cout << c.real() << " "; } std::cout << "]" << std::endl;
    }

    // 4. Scores x V
    std::cout << "  Evaluating coefficients x V..." << std::endl;
    auto ctAttnOut = EvalMatMulCtCtScoresV(cc, ctCoeffs, ctV, L, D);

    // Intermediate tracing for Attention context output
    if (secretKey) {
        Plaintext pt;
        cc->Decrypt(secretKey, ctAttnOut, &pt); pt->SetLength(L * D);
        auto val = pt->GetCKKSPackedValue();
        std::cout << "    [DEBUG ctAttnOut] level: " << ctAttnOut->GetLevel() << ", values: [ ";
        for (auto& c : val) { std::cout << c.real() << " "; } std::cout << "]" << std::endl;
    }

    // 5. Output projection W_O
    std::cout << "  Evaluating output projection W_O..." << std::endl;
    auto ctOut = EvalMatMulCtPtRect(cc, ctAttnOut, W_O, L, D, D);
    ctOut->SetSlots(slots);
    return ctOut;
}

/**
 * ============================================================================
 * 6. TEST HARNESS & GROUND TRUTH GENERATOR
 * ============================================================================
 */

int main() {
    std::cout << "=========================================================" << std::endl;
    std::cout << "GPT-2 FHE Transformer Block Complete Pipeline (CPU Only)" << std::endl;
    std::cout << "=========================================================" << std::endl;

    const uint32_t L = 2; // Sequence length
    const uint32_t D = 4; // Head/Token dimension

    // Setup parameter budget for 2-stage bootstrapped execution
    std::vector<uint32_t> levelBudget = {3, 3};
    std::vector<uint32_t> bsgsDim = {0, 0};
    uint32_t levelsAvailableAfterBootstrap = 40;

    CCParams<CryptoContextCKKSRNS> parameters;
    SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 15); // N = 32768, very fast on CPU

    ScalingTechnique rescaleTech = FLEXIBLEAUTO;
    usint dcrtBits               = 50;
    usint firstMod               = 60;
    parameters.SetScalingModSize(dcrtBits);
    parameters.SetScalingTechnique(rescaleTech);
    parameters.SetFirstModSize(firstMod);

    // Total multiplicative depth = levels needed for FFN + Bootstrapping overhead
    usint depth = levelsAvailableAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);
    parameters.SetMultiplicativeDepth(depth);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t slots = cc->GetRingDimension() / 2;
    std::cout << "CryptoContext cyclotomic order M = " << cc->GetCyclotomicOrder() 
              << ", ring dimension N = " << cc->GetRingDimension() 
              << ", slots = " << slots << std::endl;

    std::cout << "Generating keys (this will take about 1-2 minutes on CPU)..." << std::endl;
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalSumKeyGen(keyPair.secretKey);

    // Precompute rotation keys for all shifts in range [-32, 32]
    std::vector<int32_t> rotationShifts;
    for (int32_t i = -32; i <= 32; ++i) {
        if (i != 0) rotationShifts.push_back(i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotationShifts);

    // Bootstrapping setup: sparse packing mode with 8 slots for execution speedup
    uint32_t numSlotsForBoot = 8;
    cc->EvalBootstrapSetup(levelBudget, bsgsDim, numSlotsForBoot);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, numSlotsForBoot);

    std::cout << "Key generation completed!" << std::endl;

    // Initialize inputs (Q, K, V)
    std::mt19937 prng(42);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    std::vector<double> flatX(L * D);
    for (uint32_t i = 0; i < L * D; ++i) {
        flatX[i] = dist(prng);
    }

    // Weights setup for LayerNorms and Projections
    std::vector<double> gamma_1(D, 1.0), beta_1(D, 0.0);
    std::vector<double> gamma_2(D, 1.0), beta_2(D, 0.0);

    auto makeRandomMatrix = [&](uint32_t r, uint32_t c) {
        std::vector<std::vector<double>> W(r, std::vector<double>(c));
        for (uint32_t i = 0; i < r; ++i) {
            for (uint32_t j = 0; j < c; ++j) {
                W[i][j] = dist(prng);
            }
        }
        return W;
    };

    auto W_Q = makeRandomMatrix(D, D);
    auto W_K = makeRandomMatrix(D, D);
    auto W_V = makeRandomMatrix(D, D);
    auto W_O = makeRandomMatrix(D, D);
    auto W_ffn1 = makeRandomMatrix(D, 4 * D);
    auto W_ffn2 = makeRandomMatrix(4 * D, D);

    // Encrypt input vector
    Plaintext ptX = cc->MakeCKKSPackedPlaintext(flatX, 1, 0, nullptr, numSlotsForBoot);
    ptX->SetLength(numSlotsForBoot);
    auto ctX = cc->Encrypt(keyPair.publicKey, ptX);

    // ============================================================================
    // PYTORCH GROUND TRUTH CALCULATIONS (For tracing verification)
    // ============================================================================
    std::cout << "\nCalculating ground truth (PyTorch reference) first..." << std::endl;
    double ln_eps = 1e-5;
    
    // 1. LayerNorm 1
    std::vector<double> refLN1(L * D);
    for (uint32_t i = 0; i < L; ++i) {
        double sum = 0.0;
        for (uint32_t j = 0; j < D; ++j) sum += flatX[i * D + j];
        double mean = sum / D;
        double var = 0.0;
        for (uint32_t j = 0; j < D; ++j) var += (flatX[i * D + j] - mean) * (flatX[i * D + j] - mean);
        var /= D;
        double inv_std = 1.0 / std::sqrt(var + ln_eps);
        for (uint32_t j = 0; j < D; ++j) {
            refLN1[i * D + j] = gamma_1[j] * (flatX[i * D + j] - mean) * inv_std + beta_1[j];
        }
    }

    // 2. Projections Q, K, V
    std::vector<double> refQ(L * D, 0.0), refK(L * D, 0.0), refV(L * D, 0.0);
    for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t j = 0; j < D; ++j) {
            for (uint32_t k = 0; k < D; ++k) {
                refQ[i * D + j] += refLN1[i * D + k] * W_Q[k][j];
                refK[i * D + j] += refLN1[i * D + k] * W_K[k][j];
                refV[i * D + j] += refLN1[i * D + k] * W_V[k][j];
            }
        }
    }

    // 3. Attention scores + Softmax
    std::vector<double> refScores(L * L, 0.0);
    for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t j = 0; j < L; ++j) {
            for (uint32_t k = 0; k < D; ++k) {
                refScores[i * L + j] += refQ[i * D + k] * refK[j * D + k];
            }
        }
    }

    double softmaxScale = 2.0;
    std::vector<double> refCoeffs(L * L, 0.0);
    for (uint32_t i = 0; i < L; ++i) {
        double maxVal = -1e9;
        for (uint32_t j = 0; j <= i; ++j) {
            maxVal = std::max(maxVal, refScores[i * L + j]);
        }
        double sumExp = 0.0;
        std::vector<double> rowExp(L, 0.0);
        for (uint32_t j = 0; j <= i; ++j) {
            rowExp[j] = std::exp((refScores[i * L + j] - maxVal) / softmaxScale);
            sumExp += rowExp[j];
        }
        for (uint32_t j = 0; j <= i; ++j) {
            refCoeffs[i * L + j] = rowExp[j] / sumExp;
        }
    }

    // 4. Scores x V
    std::vector<double> refAttn(L * D, 0.0);
    for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t j = 0; j < D; ++j) {
            for (uint32_t k = 0; k < L; ++k) {
                refAttn[i * D + j] += refCoeffs[i * L + k] * refV[k * D + j];
            }
        }
    }

    // 5. Output projection W_O
    std::vector<double> refAttnProj(L * D, 0.0);
    for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t j = 0; j < D; ++j) {
            for (uint32_t k = 0; k < D; ++k) {
                refAttnProj[i * D + j] += refAttn[i * D + k] * W_O[k][j];
            }
        }
    }

    // 6. Residual 1
    std::vector<double> refRes1(L * D);
    for (uint32_t i = 0; i < L * D; ++i) {
        refRes1[i] = flatX[i] + refAttnProj[i];
    }

    // 7. LayerNorm 2
    std::vector<double> refLN2(L * D);
    for (uint32_t i = 0; i < L; ++i) {
        double sum = 0.0;
        for (uint32_t j = 0; j < D; ++j) sum += refRes1[i * D + j];
        double mean = sum / D;
        double var = 0.0;
        for (uint32_t j = 0; j < D; ++j) var += (refRes1[i * D + j] - mean) * (refRes1[i * D + j] - mean);
        var /= D;
        double inv_std = 1.0 / std::sqrt(var + ln_eps);
        for (uint32_t j = 0; j < D; ++j) {
            refLN2[i * D + j] = gamma_2[j] * (refRes1[i * D + j] - mean) * inv_std + beta_2[j];
        }
    }

    // 8. FFN 1
    std::vector<double> refFFN1(L * 4 * D, 0.0);
    for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t j = 0; j < 4 * D; ++j) {
            for (uint32_t k = 0; k < D; ++k) {
                refFFN1[i * 4 * D + j] += refLN2[i * D + k] * W_ffn1[k][j];
            }
        }
    }

    // 9. GELU
    std::vector<double> refGELU(L * 4 * D);
    for (uint32_t i = 0; i < L * 4 * D; ++i) {
        double val = refFFN1[i];
        refGELU[i] = 0.5 * val * (1.0 + std::erf(val / std::sqrt(2.0)));
    }

    // 10. FFN 2
    std::vector<double> refFFN2(L * D, 0.0);
    for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t j = 0; j < D; ++j) {
            for (uint32_t k = 0; k < 4 * D; ++k) {
                refFFN2[i * D + j] += refGELU[i * 4 * D + k] * W_ffn2[k][j];
            }
        }
    }

    // 11. Residual 2 (final block output)
    std::vector<double> refRes2(L * D);
    for (uint32_t i = 0; i < L * D; ++i) {
        refRes2[i] = refRes1[i] + refFFN2[i];
    }

    // Lambda to decrypt, print, and compare any intermediate FHE ciphertext against PyTorch reference values.
    auto debugDecrypt = [&](const std::string& name, Ciphertext<DCRTPoly> ct, const std::vector<double>& truth, uint32_t len) {
        Plaintext pt;
        cc->Decrypt(keyPair.secretKey, ct, &pt);
        pt->SetLength(len);
        auto vals = pt->GetCKKSPackedValue();
        std::cout << "\n[DEBUG " << name << "] level: " << ct->GetLevel() << std::endl;
        double maxErr = 0.0;
        for (uint32_t i = 0; i < len; ++i) {
            double fheVal = vals[i].real();
            double err = std::abs(fheVal - truth[i]);
            maxErr = std::max(maxErr, err);
            std::cout << "    Slot " << i << ": FHE=" << fheVal << " | Truth=" << truth[i] << " | Err=" << err << std::endl;
        }
        std::cout << "    Max Error for " << name << ": " << maxErr << std::endl;
    };

    // ============================================================================
    // PIPELINE EXECUTION WITH STEP-BY-STEP VERIFICATION
    // ============================================================================

    // ----------------------------------------------------------------------------
    // STAGE 1: LayerNorm 1 + Self-Attention + Residual 1
    // ----------------------------------------------------------------------------
    std::cout << "\n[Stage 1] Evaluating LayerNorm 1..." << std::endl;
    double ln_z0 = 0.2;
    int ln_newton = 2;
    auto ctLN1 = EvalRowWiseLayerNorm(cc, ctX, L, D, gamma_1, beta_1, ln_eps, ln_z0, ln_newton);
    debugDecrypt("LayerNorm 1", ctLN1, refLN1, L * D);

    std::cout << "\n[Stage 1] Evaluating Causal Attention..." << std::endl;
    int df_soft = 1, dg_soft = 0, r_soft = 3, gold_soft = 3;
    // Pass the secret key to enable nested decrypt tracing inside the attention calculations
    auto ctAttn = EvalCausalAttention(cc, ctLN1, L, D, W_Q, W_K, W_V, W_O, softmaxScale, df_soft, dg_soft, r_soft, gold_soft, keyPair.secretKey);
    debugDecrypt("Attention Projection Output", ctAttn, refAttnProj, L * D);

    std::cout << "\n[Stage 1] Adding residual 1..." << std::endl;
    uint32_t maxLvlRes1 = std::max(ctX->GetLevel(), ctAttn->GetLevel());
    auto ctRes1 = cc->EvalAdd(levelReduceTo(cc, ctX, maxLvlRes1), levelReduceTo(cc, ctAttn, maxLvlRes1));
    debugDecrypt("Residual 1 Output", ctRes1, refRes1, L * D);

    // ----------------------------------------------------------------------------
    // STAGE 2: Bootstrap 1 (Resets noise before LayerNorm 2 & FFN)
    // ----------------------------------------------------------------------------
    std::cout << "\n[Stage 2] Bootstrapping stage 1 output..." << std::endl;
    auto ctBoot1 = cc->EvalBootstrap(ctRes1);
    std::cout << "  Bootstrapped remaining levels: " << depth - ctBoot1->GetLevel() << " / " << depth << std::endl;
    debugDecrypt("Bootstrapped Residual 1 Output", ctBoot1, refRes1, L * D);

    // ----------------------------------------------------------------------------
    // STAGE 3: LayerNorm 2 + FFN (Piecewise GELU) + Residual 2
    // ----------------------------------------------------------------------------
    std::cout << "\n[Stage 3] Evaluating LayerNorm 2..." << std::endl;
    // Reduced to 2 iterations (from 3) to save 3 multiplicative levels, ensuring the subsequent GELU activation doesn't exhaust the post-bootstrap depth limit.
    int ln_iters = 2;
    auto ctLN2 = EvalRowWiseLayerNorm(cc, ctBoot1, L, D, gamma_2, beta_2, ln_eps, ln_z0, ln_iters);
    debugDecrypt("LayerNorm 2 Output", ctLN2, refLN2, L * D);

    std::cout << "\n[Stage 3] Evaluating FFN 1 projection..." << std::endl;
    auto ctFFN1 = EvalMatMulCtPtRect(cc, ctLN2, W_ffn1, L, D, 4 * D);
    debugDecrypt("FFN 1 Output (Pre-GELU)", ctFFN1, refFFN1, L * 4 * D);

    std::cout << "\n[Stage 3] Evaluating Piecewise GELU..." << std::endl;
    int df_gelu = 1, dg_gelu = 1;
    auto ctGELU = approx_gelu_piecewise(cc, ctFFN1, df_gelu, dg_gelu);
    debugDecrypt("GELU Output", ctGELU, refGELU, L * 4 * D);

    std::cout << "\n[Stage 3] Evaluating FFN 2 projection..." << std::endl;
    auto ctFFN2 = EvalMatMulCtPtRect(cc, ctGELU, W_ffn2, L, 4 * D, D);
    debugDecrypt("FFN 2 Output", ctFFN2, refFFN2, L * D);

    std::cout << "\n[Stage 3] Adding residual 2..." << std::endl;
    uint32_t maxLvlRes2 = std::max(ctBoot1->GetLevel(), ctFFN2->GetLevel());
    auto ctRes2 = cc->EvalAdd(levelReduceTo(cc, ctBoot1, maxLvlRes2), levelReduceTo(cc, ctFFN2, maxLvlRes2));
    debugDecrypt("Residual 2 Output (Final)", ctRes2, refRes2, L * D);

    return 0;
}
