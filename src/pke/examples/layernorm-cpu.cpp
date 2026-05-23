#include "openfhe.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace lbcrypto;

/**
 * Approximate LayerNorm using OpenFHE CPU operations.
 *
 * Translates approximations/layer_norm.py into FHE:
 *   1. Mean:  mu = EvalSum(x) / n
 *   2. Variance: sigma^2 = EvalSum((x - mu)^2) / n
 *   3. Inverse sqrt via Taylor expansion + Newton iterations
 *   4. Output: (x - mu) * inv_sqrt * weight + bias
 *
 * Since CKKS slots are packed horizontally, "mean over n slots" is done
 * with cc->EvalSum(ct, n) which adds all slots together, followed by
 * EvalMult(ct, 1.0/n) to divide.
 */

// Taylor expansion initial seed for 1/sqrt(z) around z0
// c0 =  1 / sqrt(z0)
// c1 = -1 / (2 * sqrt(z0^3))
// c2 =  3 / (8 * sqrt(z0^5))
// c3 = -5 / (16 * sqrt(z0^7))
struct TaylorCoeffs {
    double c0, c1, c2, c3;
};

TaylorCoeffs computeTaylorCoeffs(double z0) {
    TaylorCoeffs t;
    t.c0 =  1.0 / std::sqrt(z0);
    t.c1 =  1.0 / (2.0 * std::sqrt(z0 * z0 * z0));
    t.c2 =  3.0 / (8.0 * std::sqrt(z0 * z0 * z0 * z0 * z0));
    t.c3 =  5.0 / (16.0 * std::sqrt(z0 * z0 * z0 * z0 * z0 * z0 * z0));
    return t;
}

Ciphertext<DCRTPoly> approxLayerNorm(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> x,
    uint32_t n,                          // number of active slots (normalized_shape)
    const std::vector<double>& weight,   // gamma
    const std::vector<double>& bias,     // beta
    double eps,
    double z0,
    int newton_iters
) {
    // ---------- Pre-scaling for stability ----------
    // We scale the input ciphertext x by k = 1 / sqrt(z0).
    // This scales the expected variance to ~1.0, allowing us to perform
    // the Taylor expansion around z0_scaled = 1.0.
    // This completely avoids massive polynomial coefficients when z0 is tiny/huge.
    double k = 1.0 / std::sqrt(z0);
    double eps_scaled = eps / z0;
    double z0_scaled = 1.0;

    auto x_scaled = cc->EvalMult(x, k);

    // ---------- Step 1: Mean ----------
    // sum_x = EvalSum(x_scaled, n) -> every slot holds sum of all n slots
    auto sum_x = cc->EvalSum(x_scaled, n);
    // mu = sum_x / n
    auto mu = cc->EvalMult(sum_x, 1.0 / static_cast<double>(n));

    // ---------- Step 2: x_centered = x - mu ----------
    auto x_centered = cc->EvalSub(x_scaled, mu);

    // ---------- Step 3: Variance ----------
    // x_centered_sq = x_centered^2
    auto x_centered_sq = cc->EvalMult(x_centered, x_centered);
    // sum_sq = EvalSum(x_centered_sq, n)
    auto sum_sq = cc->EvalSum(x_centered_sq, n);
    // sigma_sq = sum_sq / n
    auto sigma_sq = cc->EvalMult(sum_sq, 1.0 / static_cast<double>(n));
    // z = sigma_sq + eps_scaled
    auto z = cc->EvalAdd(sigma_sq, eps_scaled);

    // ---------- Step 4: Taylor expansion of 1/sqrt(z) around z0_scaled = 1.0 ----------
    TaylorCoeffs tc = computeTaylorCoeffs(z0_scaled);

    // z_diff = z - z0_scaled
    auto z_diff = cc->EvalSub(z, z0_scaled);
    // z_diff^2
    auto z_diff_sq = cc->EvalMult(z_diff, z_diff);
    // z_diff^3
    auto z_diff_cub = cc->EvalMult(z_diff_sq, z_diff);

    // y = c0 - c1*z_diff + c2*z_diff^2 - c3*z_diff^3
    auto y = cc->EvalMult(z_diff, tc.c1);
    y = cc->EvalSub(tc.c0, y);

    auto term2 = cc->EvalMult(z_diff_sq, tc.c2);
    y = cc->EvalAdd(y, term2);

    auto term3 = cc->EvalMult(z_diff_cub, tc.c3);
    y = cc->EvalSub(y, term3);

    // ---------- Step 5: Newton iterations for 1/sqrt(z) ----------
    // y_{i+1} = y_i * (3 - z * y_i^2) / 2
    // Each iteration costs ~4 multiplicative depths:
    //   y_sq = y*y (1), z_y_sq = z*y_sq (1), y*inner (1), *0.5 (1)
    for (int i = 0; i < newton_iters; ++i) {
        auto y_sq = cc->EvalMult(y, y);
        auto z_y_sq = cc->EvalMult(z, y_sq);
        auto inner = cc->EvalSub(3.0, z_y_sq);
        auto y_inner = cc->EvalMult(y, inner);
        y = cc->EvalMult(y_inner, 0.5);
    }

    // ---------- Step 6: x_norm = x_centered * y ----------
    auto x_norm = cc->EvalMult(x_centered, y);

    // ---------- Step 7: Apply affine transform: weight * x_norm + bias ----------
    Plaintext ptWeight = cc->MakeCKKSPackedPlaintext(weight);
    Plaintext ptBias = cc->MakeCKKSPackedPlaintext(bias);

    auto result = cc->EvalMult(x_norm, ptWeight);
    result = cc->EvalAdd(result, ptBias);

    return result;
}

int main() {
    std::cout << "LayerNorm Approximation Test (CPU Only)" << std::endl;

    // LayerNorm parameters
    const uint32_t n = 8;          // normalized_shape
    const double eps = 1e-5;
    const int newton_iters = 2;    // Newton refinement iterations

    // Depth budget:
    //  - mean: 1 mult (scaling by 1/n)
    //  - x_centered_sq: 1 mult
    //  - variance: 1 mult (scaling by 1/n)
    //  - z_diff^2: 1 mult, z_diff^3: 1 mult
    //  - Taylor terms: 3 mults (c1*z_diff, c2*z_diff^2, c3*z_diff^3)
    //  - Newton: 2 iters × 4 mults each = 8
    //  - x_norm: 1 mult
    //  - final affine: 1 mult
    // Total ≈ 18, use 30 for safety margin
    const int mult_depth = 30;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(mult_depth);
    parameters.SetScalingModSize(50);
    parameters.SetFirstModSize(60);
    parameters.SetBatchSize(n);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    std::cout << "Generating keys..." << std::endl;
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalSumKeyGen(keyPair.secretKey);

    // Weight = 1.0 (gamma), Bias = 0.0 (beta) for testing
    std::vector<double> weight(n, 1.0);
    std::vector<double> bias(n, 0.0);

    // NOTE: z0 must be set close to the expected variance of the input.
    // In neural network inference, activation statistics are known at
    // deployment time, so z0 is calibrated per-layer. The Taylor expansion
    // of 1/sqrt(z) around z0 only converges when |z - z0| < z0.

    // Helper lambda to compute ground truth and compare
    auto runTest = [&](const std::string& name,
                       const std::vector<std::complex<double>>& input,
                       const std::vector<double>& w,
                       const std::vector<double>& b,
                       double test_z0) {
        Plaintext pt = cc->MakeCKKSPackedPlaintext(input);
        auto ct = cc->Encrypt(keyPair.publicKey, pt);

        std::cout << "\n--- " << name << " ---" << std::endl;
        std::cout << "  z0 = " << test_z0 << std::endl;
        std::cout << "  Evaluating LayerNorm..." << std::endl;
        auto result = approxLayerNorm(cc, ct, n, w, b, eps, test_z0, newton_iters);

        Plaintext ptRes;
        cc->Decrypt(keyPair.secretKey, result, &ptRes);
        ptRes->SetLength(n);

        // Compute ground truth
        double sum = 0;
        for (auto& v : input) sum += v.real();
        double mean = sum / n;
        double var = 0;
        for (auto& v : input) var += (v.real() - mean) * (v.real() - mean);
        var /= n;
        double inv_std = 1.0 / std::sqrt(var + eps);
        std::cout << "  True variance = " << var << std::endl;

        auto res_vals = ptRes->GetCKKSPackedValue();
        double max_err = 0;
        for (uint32_t i = 0; i < n; ++i) {
            double x_norm = (input[i].real() - mean) * inv_std;
            double truth = w[i] * x_norm + b[i];
            double fhe_val = res_vals[i].real();
            double err = std::abs(fhe_val - truth);
            max_err = std::max(max_err, err);
            std::cout << "  Slot " << i << ": FHE=" << fhe_val
                      << " | Truth=" << truth << " | Err=" << err << std::endl;
        }
        std::cout << "  Max Error: " << max_err << std::endl;
        std::cout << "  " << (max_err < 0.05 ? "SUCCESS" : "WARNING: large error") << std::endl;
    };

    // Test 1: Sequential values [1..8], variance = 5.25
    runTest("Test 1: Sequential [1..8]",
            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
            weight, bias, 5.25);

    // Test 2: Mixed negative/positive, variance ≈ 13.5
    runTest("Test 2: Mixed negative/positive",
            {-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 7.0},
            weight, bias, 14.0);

    // Test 3: Realistic neural net activations (variance near 1.0)
    runTest("Test 3: Realistic activations (var~1)",
            {-1.2, -0.8, -0.3, 0.1, 0.4, 0.9, 1.3, 1.6},
            weight, bias, 1.0);

    // Test 4: Very small values [0.001..0.008], variance ≈ 5.25e-6
    runTest("Test 4: Very small values",
            {0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008},
            weight, bias, 5.0e-6);

    // Test 5: Custom weight and bias
    {
        std::vector<double> w = {0.5, 1.0, 1.5, 2.0, 0.5, 1.0, 1.5, 2.0};
        std::vector<double> b = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
        runTest("Test 5: Custom weight/bias",
                {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                w, b, 5.25);
    }

    // Test 6: Very large negative values (near -1000) with small variance
    runTest("Test 6: Large negative, small var",
            {-1004.0, -1003.0, -1002.0, -1001.0, -1000.0, -999.0, -998.0, -997.0},
            weight, bias, 5.25);

    // Test 7: Very large negative values (around -1000) with extremely large variance
    runTest("Test 7: Large negative, large var",
            {-2000.0, -1500.0, -1200.0, -1000.0, -800.0, -500.0, -200.0, -100.0},
            weight, bias, 371093.75);

    std::cout << "\nLayerNorm Approximation Test Completed." << std::endl;
    return 0;
}
