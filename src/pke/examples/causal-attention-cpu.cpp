#include "openfhe.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

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
Ciphertext<DCRTPoly> approxMax2(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> a,
    Ciphertext<DCRTPoly> b,
    double scale,
    int df,
    int dg
) {
    uint32_t max_input_lvl = std::max(a->GetLevel(), b->GetLevel());
    auto a_aligned = levelReduceTo(cc, a, max_input_lvl);
    auto b_aligned = levelReduceTo(cc, b, max_input_lvl);

    auto diff = cc->EvalSub(a_aligned, b_aligned);
    auto diff_scaled = cc->EvalMult(diff, 1.0 / scale);
    
    auto sign_val = Eval_accelerated_sign(cc, diff_scaled, df, dg);

    uint32_t sign_lvl = sign_val->GetLevel();
    auto diff_leveled = levelReduceTo(cc, diff, sign_lvl);
    auto abs_diff = cc->EvalMult(diff_leveled, sign_val);

    auto sum = cc->EvalAdd(a_aligned, b_aligned);

    uint32_t abs_diff_lvl = abs_diff->GetLevel();
    auto sum_leveled = levelReduceTo(cc, sum, abs_diff_lvl);

    auto sum_abs = cc->EvalAdd(sum_leveled, abs_diff);
    return cc->EvalMult(sum_abs, 0.5);
}

// Helper: Replicate slot 0 to the first 'count' slots, preventing wrap-around corruption
Ciphertext<DCRTPoly> ReplicateSlot0(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> ct,
    uint32_t count
) {
    std::vector<double> mask0(ct->GetSlots(), 0.0);
    mask0[0] = 1.0;
    Plaintext ptMask0 = cc->MakeCKKSPackedPlaintext(mask0);
    Ciphertext<DCRTPoly> result = cc->EvalMult(ct, ptMask0);

    uint32_t currentReplicas = 1;
    while (currentReplicas < count) {
        uint32_t shift = currentReplicas;
        auto rotated = cc->EvalRotate(result, -static_cast<int32_t>(shift));
        result = cc->EvalAdd(result, rotated);
        currentReplicas *= 2;
    }
    return result;
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
    // Replicate slot 0 max value to all n slots to make subtraction correct
    return ReplicateSlot0(cc, current_max, n);
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
    // 1. Find max of x (replicated to first n slots)
    auto x_max = fheMax(cc, x, n, scale, df, dg);

    // 2. Subtract max (x_shifted = x - x_max)
    uint32_t max_lvl = x_max->GetLevel();
    auto x_leveled = levelReduceTo(cc, x, max_lvl);
    auto x_shifted = cc->EvalSub(x_leveled, x_max);

    // 3. Evaluate exponential
    auto exp_x = approxExp(cc, x_shifted, r);

    // 4. Sum of exponentials (sum_exp replicated to all slots by EvalSum)
    auto sum_exp = cc->EvalSum(exp_x, n);

    // 5. Goldschmidt division
    auto out = goldschmidtDivision(cc, exp_x, sum_exp, n, goldschmidt_iters);

    return out;
}

// 2. Ciphertext-Ciphertext Matrix Multiplication (Q x K^T)
Ciphertext<DCRTPoly> EvalMatMulCtCtQTKT(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> ctQ,
    Ciphertext<DCRTPoly> ctK,
    uint32_t L, uint32_t D
) {
    uint32_t slots = ctQ->GetSlots();
    std::vector<double> maskD(slots, 0.0);
    std::fill(maskD.begin(), maskD.begin() + D, 1.0);
    Plaintext ptMaskD = cc->MakeCKKSPackedPlaintext(maskD);

    std::vector<Ciphertext<DCRTPoly>> dotProducts(L * L);

    for (uint32_t i = 0; i < L; ++i) {
        Ciphertext<DCRTPoly> ctQ_i = (i * D == 0) ? ctQ : cc->EvalRotate(ctQ, i * D);
        ctQ_i = cc->EvalMult(ctQ_i, ptMaskD);

        for (uint32_t j = 0; j < L; ++j) {
            Ciphertext<DCRTPoly> ctK_j = (j * D == 0) ? ctK : cc->EvalRotate(ctK, j * D);
            ctK_j = cc->EvalMult(ctK_j, ptMaskD);

            Ciphertext<DCRTPoly> prod = cc->EvalMult(ctQ_i, ctK_j);
            Ciphertext<DCRTPoly> sum = cc->EvalSum(prod, D);

            dotProducts[i * L + j] = sum;
        }
    }

    return cc->EvalMerge(dotProducts);
}

// 3. Ciphertext-Ciphertext Matrix Multiplication (Scores x V)
Ciphertext<DCRTPoly> EvalMatMulCtCtScoresV(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> ctScores,
    Ciphertext<DCRTPoly> ctV,
    uint32_t L, uint32_t D
) {
    uint32_t slots = ctScores->GetSlots();
    std::vector<double> maskD(slots, 0.0);
    std::fill(maskD.begin(), maskD.begin() + D, 1.0);
    Plaintext ptMaskD = cc->MakeCKKSPackedPlaintext(maskD);

    std::vector<Ciphertext<DCRTPoly>> rowResults(L);

    for (uint32_t i = 0; i < L; ++i) {
        Ciphertext<DCRTPoly> ctRowSum;
        bool isFirst = true;

        for (uint32_t k = 0; k < L; ++k) {
            Ciphertext<DCRTPoly> ctScalar = (i * L + k == 0) ? ctScores : cc->EvalRotate(ctScores, i * L + k);
            Ciphertext<DCRTPoly> ctScalarRep = ReplicateSlot0(cc, ctScalar, D);

            Ciphertext<DCRTPoly> ctV_k = (k * D == 0) ? ctV : cc->EvalRotate(ctV, k * D);
            ctV_k = cc->EvalMult(ctV_k, ptMaskD);

            // Align levels before multiplication
            uint32_t scalarLvl = ctScalarRep->GetLevel();
            ctV_k = levelReduceTo(cc, ctV_k, scalarLvl);

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

    Ciphertext<DCRTPoly> result;
    bool isFirst = true;
    for (uint32_t i = 0; i < L; ++i) {
        std::vector<double> maskRow(slots, 0.0);
        std::fill(maskRow.begin() + i * D, maskRow.begin() + (i + 1) * D, 1.0);
        Plaintext ptMaskRow = cc->MakeCKKSPackedPlaintext(maskRow);

        Ciphertext<DCRTPoly> shiftedRow = (i * D == 0) ? rowResults[i] : cc->EvalRotate(rowResults[i], -static_cast<int32_t>(i * D));
        Ciphertext<DCRTPoly> maskedRow = cc->EvalMult(shiftedRow, ptMaskRow);

        if (isFirst) {
            result = maskedRow;
            isFirst = false;
        } else {
            result = cc->EvalAdd(result, maskedRow);
        }
    }

    return result;
}

int main() {
    std::cout << "GPT-2 Causal Self-Attention FHE Block Test (CPU Only)" << std::endl;

    const uint32_t L = 2; // Sequence length
    const uint32_t D = 4; // Head dimension

    // Softmax parameters
    const int r = 3;                 // Exp accuracy parameter
    const int goldschmidt_iters = 3; // Goldschmidt division iterations
    const int actual_df = 1;
    const int actual_dg = 0;
    const double softmaxScale = 4.0;

    // Depth calculation:
    // 2 (QTKT) + 1 (masking row) + 7 (fheMax) + 4 (exp) + 4 (division) + 4 (Scores x V) = 22 levels
    // Set to 40 to leave plenty of levels of budget for decryption precision
    const int mult_depth = 40;

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

    uint32_t slots = cc->GetRingDimension() / 2;
    std::cout << "CryptoContext actual cyclotomic order M = " << cc->GetCyclotomicOrder() 
              << ", ring dimension N = " << cc->GetRingDimension() 
              << ", slots = " << slots << std::endl;

    std::cout << "Generating keys (optimized rotation keys list)..." << std::endl;
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalSumKeyGen(keyPair.secretKey);

    // Optimized list of rotation shifts
    std::vector<int32_t> rotationShifts = {1, 2, 3, 4, -1, -2, -3, -4};
    cc->EvalRotateKeyGen(keyPair.secretKey, rotationShifts);

    // Initialize inputs (Q, K, V)
    std::mt19937 prng(42);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    std::vector<double> flatQ(L * D), flatK(L * D), flatV(L * D);
    for (uint32_t i = 0; i < L * D; ++i) {
        flatQ[i] = dist(prng);
        flatK[i] = dist(prng);
        flatV[i] = dist(prng);
    }

    // Print input matrices
    std::cout << "\nQ Matrix:\n";
    for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t j = 0; j < D; ++j) std::cout << flatQ[i * D + j] << " ";
        std::cout << "\n";
    }
    std::cout << "\nK Matrix:\n";
    for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t j = 0; j < D; ++j) std::cout << flatK[i * D + j] << " ";
        std::cout << "\n";
    }
    std::cout << "\nV Matrix:\n";
    for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t j = 0; j < D; ++j) std::cout << flatV[i * D + j] << " ";
        std::cout << "\n";
    }

    auto ctQ = cc->Encrypt(keyPair.publicKey, cc->MakeCKKSPackedPlaintext(flatQ));
    auto ctK = cc->Encrypt(keyPair.publicKey, cc->MakeCKKSPackedPlaintext(flatK));
    auto ctV = cc->Encrypt(keyPair.publicKey, cc->MakeCKKSPackedPlaintext(flatV));

    std::cout << "\nEvaluating Q x K^T..." << std::endl;
    auto ctScores = EvalMatMulCtCtQTKT(cc, ctQ, ctK, L, D);
    std::cout << "ctScores level: " << ctScores->GetLevel() << std::endl;

    // Causal Masking plaintext formulation
    // j > i is masked by -50.0 (practically -inf)
    std::vector<double> causalMask(slots, 0.0);
    for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t j = 0; j < L; ++j) {
            if (j > i) {
                causalMask[i * L + j] = -50.0;
            }
        }
    }
    Plaintext ptCausalMask = cc->MakeCKKSPackedPlaintext(causalMask);

    std::cout << "Applying causal mask..." << std::endl;
    auto ctScoresMasked = cc->EvalAdd(ctScores, ptCausalMask);
    std::cout << "ctScoresMasked level: " << ctScoresMasked->GetLevel() << std::endl;

    std::cout << "Evaluating row-wise softmax..." << std::endl;
    Ciphertext<DCRTPoly> ctCoeffs;
    std::vector<Ciphertext<DCRTPoly>> rowResults(L);

    for (uint32_t i = 0; i < L; ++i) {
        std::cout << "  Processing row " << i << " / " << L << "..." << std::endl;
        // 1. Extract row i to slot [0..i] (only active elements)
        Ciphertext<DCRTPoly> ctRow = (i * L == 0) ? ctScores : cc->EvalRotate(ctScores, i * L);
        std::vector<double> maskActive(slots, 0.0);
        std::fill(maskActive.begin(), maskActive.begin() + i + 1, 1.0);
        Plaintext ptMaskActive = cc->MakeCKKSPackedPlaintext(maskActive);
        ctRow = cc->EvalMult(ctRow, ptMaskActive);
        std::cout << "    ctRow level: " << ctRow->GetLevel() << std::endl;

        // 2. Compute softmax on row of size i + 1
        Ciphertext<DCRTPoly> ctRowSoftmax = approxSoftmax(cc, ctRow, i + 1, softmaxScale, actual_df, actual_dg, r, goldschmidt_iters);
        std::cout << "    ctRowSoftmax level: " << ctRowSoftmax->GetLevel() << std::endl;

        // 3. Shift back
        Ciphertext<DCRTPoly> ctRowBack = (i * L == 0) ? ctRowSoftmax : cc->EvalRotate(ctRowSoftmax, -static_cast<int32_t>(i * L));
        std::vector<double> maskRowi(slots, 0.0);
        std::fill(maskRowi.begin() + i * L, maskRowi.begin() + i * L + i + 1, 1.0);
        Plaintext ptMaskRowi = cc->MakeCKKSPackedPlaintext(maskRowi);
        ctRowBack = cc->EvalMult(ctRowBack, ptMaskRowi);
        std::cout << "    ctRowBack level: " << ctRowBack->GetLevel() << std::endl;

        rowResults[i] = ctRowBack;
    }

    // Align all rows to the maximum level among them
    uint32_t maxLvl = 0;
    for (uint32_t i = 0; i < L; ++i) {
        maxLvl = std::max<uint32_t>(maxLvl, rowResults[i]->GetLevel());
    }
    std::cout << "Aligning all rows to level " << maxLvl << std::endl;
    for (uint32_t i = 0; i < L; ++i) {
        rowResults[i] = levelReduceTo(cc, rowResults[i], maxLvl);
        if (i == 0) {
            ctCoeffs = rowResults[i];
        } else {
            ctCoeffs = cc->EvalAdd(ctCoeffs, rowResults[i]);
        }
    }
    std::cout << "ctCoeffs level: " << ctCoeffs->GetLevel() << std::endl;

    // Decrypt intermediate ctScores
    try {
        Plaintext ptScoresDec;
        cc->Decrypt(keyPair.secretKey, ctScores, &ptScoresDec);
        ptScoresDec->SetLength(L * L);
        std::cout << "ctScores decrypted successfully: [ ";
        for (uint32_t i = 0; i < L * L; ++i) {
            std::cout << ptScoresDec->GetCKKSPackedValue()[i].real() << " ";
        }
        std::cout << "]" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "ctScores decryption FAILED: " << e.what() << std::endl;
    }

    // Decrypt intermediate ctCoeffs
    try {
        Plaintext ptCoeffsDec;
        cc->Decrypt(keyPair.secretKey, ctCoeffs, &ptCoeffsDec);
        ptCoeffsDec->SetLength(L * L);
        std::cout << "ctCoeffs decrypted successfully: [ ";
        for (uint32_t i = 0; i < L * L; ++i) {
            std::cout << ptCoeffsDec->GetCKKSPackedValue()[i].real() << " ";
        }
        std::cout << "]" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "ctCoeffs decryption FAILED: " << e.what() << std::endl;
    }

    std::cout << "\nEvaluating coefficients x V..." << std::endl;
    auto ctOut = EvalMatMulCtCtScoresV(cc, ctCoeffs, ctV, L, D);
    std::cout << "ctOut level: " << ctOut->GetLevel() << std::endl;

    // Decrypt and compare
    Plaintext ptOut;
    std::vector<std::complex<double>> resOut;
    try {
        cc->Decrypt(keyPair.secretKey, ctOut, &ptOut);
        ptOut->SetLength(L * D);
        resOut = ptOut->GetCKKSPackedValue();
        std::cout << "ctOut decrypted successfully: [ ";
        for (uint32_t i = 0; i < L * D; ++i) {
            std::cout << resOut[i].real() << " ";
        }
        std::cout << "]" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "ctOut decryption FAILED: " << e.what() << std::endl;
    }

    // ----------------------------------------------------
    // GROUND TRUTH CALCULATION (PyTorch style)
    // ----------------------------------------------------
    // 1. Scores = Q x K^T
    std::vector<double> expectedScores(L * L, 0.0);
    for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t j = 0; j < L; ++j) {
            double val = 0.0;
            for (uint32_t k = 0; k < D; ++k) {
                val += flatQ[i * D + k] * flatK[j * D + k];
            }
            expectedScores[i * L + j] = val;
        }
    }

    // 2. Apply Causal Mask & Softmax
    std::vector<double> expectedCoeffs(L * L, 0.0);
    for (uint32_t i = 0; i < L; ++i) {
        double maxVal = -1e9;
        std::vector<double> rowScores(L);
        for (uint32_t j = 0; j < L; ++j) {
            double val = expectedScores[i * L + j];
            if (j > i) {
                val += -50.0; // Mask
            }
            rowScores[j] = val;
            maxVal = std::max(maxVal, val);
        }

        double sumExp = 0.0;
        std::vector<double> rowExp(L);
        for (uint32_t j = 0; j < L; ++j) {
            rowExp[j] = std::exp(rowScores[j] - maxVal);
            sumExp += rowExp[j];
        }

        for (uint32_t j = 0; j < L; ++j) {
            expectedCoeffs[i * L + j] = rowExp[j] / sumExp;
        }
    }

    // 3. Expected Out = Coeffs x V
    std::vector<double> expectedOut(L * D, 0.0);
    for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t j = 0; j < D; ++j) {
            double val = 0.0;
            for (uint32_t k = 0; k < L; ++k) {
                val += expectedCoeffs[i * L + k] * flatV[k * D + j];
            }
            expectedOut[i * D + j] = val;
        }
    }

    // Print comparison
    double maxErr = 0.0;
    std::cout << "\nComparison - FHE Output vs Ground Truth Causal Self-Attention:\n";
    for (uint32_t i = 0; i < L; ++i) {
        std::cout << "Row " << i << ":\n";
        for (uint32_t j = 0; j < D; ++j) {
            double fheVal = resOut[i * D + j].real();
            double truth = expectedOut[i * D + j];
            double err = std::abs(fheVal - truth);
            maxErr = std::max(maxErr, err);
            std::cout << "  Col " << j << ": FHE=" << fheVal << " | Truth=" << truth << " | Err=" << err << "\n";
        }
    }

    std::cout << "\nMax Causal Self-Attention pipeline Error: " << maxErr << std::endl;
    if (maxErr < 0.05) {
        std::cout << "\nALL TESTS PASSED: GPT-2 Causal Self-Attention FHE Block evaluates correctly!" << std::endl;
    } else {
        std::cout << "\nFAILURE: Large error detected." << std::endl;
    }

    return 0;
}
