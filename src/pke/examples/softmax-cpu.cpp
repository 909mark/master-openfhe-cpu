#include "openfhe.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace lbcrypto;

// Helper to reduce the level of a ciphertext to a target level
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

// Helper to evaluate accelerated sign
Ciphertext<DCRTPoly> Eval_accelerated_sign(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> x,
    int df,
    int dg
) {
    // Polynomial coefficients for f4(x) and g4(x) from sign.py
    std::vector<double> coeff_f4 = {0.0, 315.0/128.0, 0.0, -420.0/128.0, 0.0, 378.0/128.0, 0.0, -180.0/128.0, 0.0, 35.0/128.0};
    std::vector<double> coeff_g4 = {0.0, 5850.0/1024.0, 0.0, -34974.0/1024.0, 0.0, 97015.0/1024.0, 0.0, -113492.0/1024.0, 0.0, 46623.0/1024.0};

    auto res = x;
    for (int i = 0; i < dg; ++i) {
        res = cc->EvalPoly(res, coeff_g4);
    }
    for (int i = 0; i < df; ++i) {
        res = cc->EvalPoly(res, coeff_f4);
    }
    return res;
}

// Compute element-wise max of two ciphertexts: max(a, b) = 0.5 * (a + b + |a - b|)
// where |a - b| = (a - b) * sign(a - b)
Ciphertext<DCRTPoly> approxMax2(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> a,
    Ciphertext<DCRTPoly> b,
    double scale,
    int df,
    int dg
) {
    // Align input levels
    uint32_t max_input_lvl = std::max(a->GetLevel(), b->GetLevel());
    auto a_aligned = levelReduceTo(cc, a, max_input_lvl);
    auto b_aligned = levelReduceTo(cc, b, max_input_lvl);

    auto diff = cc->EvalSub(a_aligned, b_aligned);
    auto diff_scaled = cc->EvalMult(diff, 1.0 / scale);
    
    auto sign_val = Eval_accelerated_sign(cc, diff_scaled, df, dg);

    // To multiply diff and sign_val, they must be at the same level
    uint32_t sign_lvl = sign_val->GetLevel();
    auto diff_leveled = levelReduceTo(cc, diff, sign_lvl);
    auto abs_diff = cc->EvalMult(diff_leveled, sign_val);

    // Compute sum = a + b
    auto sum = cc->EvalAdd(a_aligned, b_aligned);

    // To add sum and abs_diff, they must be at the same level
    uint32_t abs_diff_lvl = abs_diff->GetLevel();
    auto sum_leveled = levelReduceTo(cc, sum, abs_diff_lvl);

    auto sum_abs = cc->EvalAdd(sum_leveled, abs_diff);
    return cc->EvalMult(sum_abs, 0.5);
}

// Computes the maximum along n slots of a ciphertext using a binary reduction tree.
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
    return current_max;
}

// Approx exp(x) for x <= 0 using: (1 + x / 2^r)^(2^r)
Ciphertext<DCRTPoly> approxExp(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> x,
    int r
) {
    double divisor = static_cast<double>(1 << r);
    auto base = cc->EvalAdd(x, divisor);
    base = cc->EvalMult(base, 1.0 / divisor);

    auto res = base;
    for (int i = 0; i < r; ++i) {
        res = cc->EvalMult(res, res);
    }
    return res;
}

// Goldschmidt division to compute Numerator / Denominator
Ciphertext<DCRTPoly> goldschmidtDivision(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> N,
    Ciphertext<DCRTPoly> D,
    uint32_t seq_len,
    int iterations
) {
    // Align levels before starting
    uint32_t max_lvl = std::max(N->GetLevel(), D->GetLevel());
    auto N_curr = levelReduceTo(cc, N, max_lvl);
    auto D_curr = levelReduceTo(cc, D, max_lvl);

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

Ciphertext<DCRTPoly> approxSoftmax(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> x,
    uint32_t n,
    double scale,
    int df,
    int dg,
    int r,
    int goldschmidt_iters
) {
    // 1. Find max of x
    auto x_max = fheMax(cc, x, n, scale, df, dg);

    // 2. Subtract max (x_shifted = x - x_max)
    // Align x to the level of x_max
    uint32_t max_lvl = x_max->GetLevel();
    auto x_leveled = levelReduceTo(cc, x, max_lvl);
    auto x_shifted = cc->EvalSub(x_leveled, x_max);

    // 3. Evaluate exponential (exp_x = approx_exp(x_shifted))
    auto exp_x = approxExp(cc, x_shifted, r);

    // 4. Sum of exponentials
    auto sum_exp = cc->EvalSum(exp_x, n);

    // 5. Goldschmidt division (exp_x / sum_exp)
    auto out = goldschmidtDivision(cc, exp_x, sum_exp, n, goldschmidt_iters);

    return out;
}

int main() {
    std::cout << "Softmax Approximation Test (CPU Only)" << std::endl;

    const uint32_t n = 8;            // Sequence length / active slots
    const int r = 7;                 // Exp parameter: (1 + x/128)^128
    const int goldschmidt_iters = 7; // Goldschmidt division iterations

    // Using df=1, dg=0 for the max computation to keep the depth reasonable.
    const int actual_df = 1;
    const int actual_dg = 0;
    const int mult_depth = 38;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(mult_depth);
    parameters.SetScalingModSize(50);
    parameters.SetFirstModSize(60);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    std::cout << "Generating keys (this may take 1-2 minutes for depth " << mult_depth << ")..." << std::endl;
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalSumKeyGen(keyPair.secretKey);
    cc->EvalRotateKeyGen(keyPair.secretKey, {1, 2, 4});

    // Helper to evaluate and compare against PyTorch/Exact Softmax
    auto runTest = [&](const std::string& name, const std::vector<std::complex<double>>& input, double test_scale) {
        std::cout << "\n--- " << name << " ---" << std::endl;
        std::cout << "Input: [ ";
        for (auto& v : input) std::cout << v.real() << " ";
        std::cout << "]" << std::endl;

        uint32_t slots = cc->GetRingDimension() / 2;
        std::vector<std::complex<double>> replicated_input(slots);
        for (uint32_t i = 0; i < slots; ++i) {
            replicated_input[i] = input[i % n];
        }

        Plaintext pt = cc->MakeCKKSPackedPlaintext(replicated_input);
        auto ct = cc->Encrypt(keyPair.publicKey, pt);

        auto decryptAndPrint = [&](const std::string& label, Ciphertext<DCRTPoly> target_ct) {
            Plaintext ptDec;
            try {
                cc->Decrypt(keyPair.secretKey, target_ct, &ptDec);
                ptDec->SetLength(n);
                auto vals = ptDec->GetCKKSPackedValue();
                std::cout << "  " << label << " (lvl " << target_ct->GetLevel() << "): [ ";
                for (uint32_t i = 0; i < n; ++i) {
                    std::cout << vals[i].real() << " ";
                }
                std::cout << "]" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "  " << label << " decryption FAILED: " << e.what() << std::endl;
            }
        };

        decryptAndPrint("Input ct", ct);

        // 1. Find max of x
        auto x_max = fheMax(cc, ct, n, test_scale, actual_df, actual_dg);
        decryptAndPrint("x_max", x_max);

        // 2. Subtract max (x_shifted = x - x_max)
        uint32_t max_lvl = x_max->GetLevel();
        auto x_leveled = levelReduceTo(cc, ct, max_lvl);
        auto x_shifted = cc->EvalSub(x_leveled, x_max);
        decryptAndPrint("x_shifted", x_shifted);

        // 3. Evaluate exponential (exp_x = approx_exp(x_shifted))
        auto exp_x = approxExp(cc, x_shifted, r);
        decryptAndPrint("exp_x", exp_x);

        // 4. Sum of exponentials
        auto sum_exp = cc->EvalSum(exp_x, n);
        decryptAndPrint("sum_exp", sum_exp);

        // 5. Goldschmidt division (exp_x / sum_exp)
        auto result = goldschmidtDivision(cc, exp_x, sum_exp, n, goldschmidt_iters);
        decryptAndPrint("Result", result);

        // Compute exact Softmax for reference
        double max_val = input[0].real();
        for (uint32_t i = 1; i < n; ++i) {
            max_val = std::max(max_val, input[i].real());
        }
        std::vector<double> exact_exp(n);
        double sum_exp_exact = 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            exact_exp[i] = std::exp(input[i].real() - max_val);
            sum_exp_exact += exact_exp[i];
        }
        std::vector<double> truth(n);
        for (uint32_t i = 0; i < n; ++i) {
            truth[i] = exact_exp[i] / sum_exp_exact;
        }

        Plaintext ptRes;
        try {
            cc->Decrypt(keyPair.secretKey, result, &ptRes);
            ptRes->SetLength(n);
            auto res_vals = ptRes->GetCKKSPackedValue();
            double max_err = 0;
            for (uint32_t i = 0; i < n; ++i) {
                double fhe_val = res_vals[i].real();
                double err = std::abs(fhe_val - truth[i]);
                max_err = std::max(max_err, err);
                std::cout << "  Slot " << i << ": FHE=" << fhe_val
                          << " | Truth=" << truth[i] << " | Err=" << err << std::endl;
            }
            std::cout << "  Max Error: " << max_err << std::endl;
            std::cout << "  " << (max_err < 0.05 ? "SUCCESS" : "WARNING: large error") << std::endl;
        } catch (...) {
            std::cout << "  Final decryption failed!" << std::endl;
        }
    };

    // Test 1: Sequential [1..8]
    runTest("Test 1: Sequential values", {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, 8.0);

    // Test 2: Uniform values
    runTest("Test 2: Uniform values", {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0}, 2.0);

    // Test 3: One large value, others small
    runTest("Test 3: One spike", {-10.0, -10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0}, 20.0);

    // Test 4: Mixed positive/negative
    runTest("Test 4: Mixed values", {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0}, 6.0);

    std::cout << "\nSoftmax Approximation Test Completed." << std::endl;
    return 0;
}
