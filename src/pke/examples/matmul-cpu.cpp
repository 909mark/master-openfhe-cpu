#include "openfhe.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

using namespace lbcrypto;

// Helper: Replicate slot 0 to the first 'count' slots
Ciphertext<DCRTPoly> ReplicateSlot0(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> ct,
    uint32_t count
) {
    // Mask to keep only slot 0 first, preventing wrap-around corruption
    std::vector<double> mask0(ct->GetSlots(), 0.0);
    mask0[0] = 1.0;
    Plaintext ptMask0 = cc->MakeCKKSPackedPlaintext(mask0);
    Ciphertext<DCRTPoly> result = cc->EvalMult(ct, ptMask0);

    // We can replicate using a tree-like structure of rotations
    uint32_t currentReplicas = 1;
    while (currentReplicas < count) {
        uint32_t shift = currentReplicas;
        auto rotated = cc->EvalRotate(result, -static_cast<int32_t>(shift));
        result = cc->EvalAdd(result, rotated);
        currentReplicas *= 2;
    }
    return result;
}

// 1. Ciphertext-Plaintext Matrix Multiplication (M = D)
Ciphertext<DCRTPoly> EvalMatMulPlainSquare(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> ctX,
    const std::vector<std::vector<double>>& W,
    uint32_t L, uint32_t D
) {
    uint32_t slots = ctX->GetSlots();
    Ciphertext<DCRTPoly> result;
    bool isFirst = true;

    for (uint32_t s = 0; s < D; ++s) {
        std::vector<double> diagLeft(slots, 0.0);
        for (uint32_t i = 0; i < L; ++i) {
            for (uint32_t j = 0; j < D - s; ++j) {
                uint32_t idx = i * D + j;
                if (idx < slots) diagLeft[idx] = W[j + s][j];
            }
        }

        std::vector<double> diagRight(slots, 0.0);
        if (s > 0) {
            for (uint32_t i = 0; i < L; ++i) {
                for (uint32_t j = D - s; j < D; ++j) {
                    uint32_t idx = i * D + j;
                    if (idx < slots) diagRight[idx] = W[j + s - D][j];
                }
            }
        }

        Ciphertext<DCRTPoly> rotLeft = (s == 0) ? ctX : cc->EvalRotate(ctX, s);
        Plaintext ptLeft = cc->MakeCKKSPackedPlaintext(diagLeft);
        auto termLeft = cc->EvalMult(rotLeft, ptLeft);

        Ciphertext<DCRTPoly> term;
        if (s > 0) {
            Ciphertext<DCRTPoly> rotRight = cc->EvalRotate(ctX, static_cast<int32_t>(s) - static_cast<int32_t>(D));
            Plaintext ptRight = cc->MakeCKKSPackedPlaintext(diagRight);
            auto termRight = cc->EvalMult(rotRight, ptRight);
            term = cc->EvalAdd(termLeft, termRight);
        } else {
            term = termLeft;
        }

        if (isFirst) {
            result = term;
            isFirst = false;
        } else {
            result = cc->EvalAdd(result, term);
        }
    }

    return result;
}

// 2. Ciphertext-Ciphertext Matrix Multiplication (Q x K^T)
// Q is L x D, K is L x D. Output A is L x L.
Ciphertext<DCRTPoly> EvalMatMulCtCtQTKT(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> ctQ,
    Ciphertext<DCRTPoly> ctK,
    uint32_t L, uint32_t D
) {
    uint32_t slots = ctQ->GetSlots();
    
    // Mask to keep first D slots
    std::vector<double> maskD(slots, 0.0);
    std::fill(maskD.begin(), maskD.begin() + D, 1.0);
    Plaintext ptMaskD = cc->MakeCKKSPackedPlaintext(maskD);

    std::vector<Ciphertext<DCRTPoly>> dotProducts(L * L);

    for (uint32_t i = 0; i < L; ++i) {
        // Extract row i of Q: rotate i * D to slot 0 and mask
        Ciphertext<DCRTPoly> ctQ_i = (i * D == 0) ? ctQ : cc->EvalRotate(ctQ, i * D);
        ctQ_i = cc->EvalMult(ctQ_i, ptMaskD);

        for (uint32_t j = 0; j < L; ++j) {
            // Extract row j of K: rotate j * D to slot 0 and mask
            Ciphertext<DCRTPoly> ctK_j = (j * D == 0) ? ctK : cc->EvalRotate(ctK, j * D);
            ctK_j = cc->EvalMult(ctK_j, ptMaskD);

            // Element-wise product
            Ciphertext<DCRTPoly> prod = cc->EvalMult(ctQ_i, ctK_j);

            // Sum all elements in slots [0, D-1] to slot 0
            Ciphertext<DCRTPoly> sum = cc->EvalSum(prod, D);

            dotProducts[i * L + j] = sum;
        }
    }

    // Merge the L*L ciphertexts (each has result in slot 0) into one ciphertext
    return cc->EvalMerge(dotProducts);
}

// 3. Ciphertext-Ciphertext Matrix Multiplication (Scores x V)
// Scores is L x L, V is L x D. Output O is L x D.
Ciphertext<DCRTPoly> EvalMatMulCtCtScoresV(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> ctScores,
    Ciphertext<DCRTPoly> ctV,
    uint32_t L, uint32_t D
) {
    uint32_t slots = ctScores->GetSlots();

    // Mask to keep first D slots
    std::vector<double> maskD(slots, 0.0);
    std::fill(maskD.begin(), maskD.begin() + D, 1.0);
    Plaintext ptMaskD = cc->MakeCKKSPackedPlaintext(maskD);

    std::vector<Ciphertext<DCRTPoly>> rowResults(L);

    for (uint32_t i = 0; i < L; ++i) {
        Ciphertext<DCRTPoly> ctRowSum;
        bool isFirst = true;

        for (uint32_t k = 0; k < L; ++k) {
            // Get scalar Scores[i][k] at slot i * L + k
            Ciphertext<DCRTPoly> ctScalar = (i * L + k == 0) ? ctScores : cc->EvalRotate(ctScores, i * L + k);
            // Replicate slot 0 of ctScalar to first D slots
            Ciphertext<DCRTPoly> ctScalarRep = ReplicateSlot0(cc, ctScalar, D);

            // Extract row k of V: rotate k * D to slot 0 and mask
            Ciphertext<DCRTPoly> ctV_k = (k * D == 0) ? ctV : cc->EvalRotate(ctV, k * D);
            ctV_k = cc->EvalMult(ctV_k, ptMaskD);

            // Multiply scalar by row vector
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

    // Pack rowResults (each is size D in slots [0, D-1]) back into a single ciphertext
    Ciphertext<DCRTPoly> result;
    bool isFirst = true;
    for (uint32_t i = 0; i < L; ++i) {
        // Construct mask to keep only slots [i*D, (i+1)*D - 1]
        std::vector<double> maskRow(slots, 0.0);
        std::fill(maskRow.begin() + i * D, maskRow.begin() + (i + 1) * D, 1.0);
        Plaintext ptMaskRow = cc->MakeCKKSPackedPlaintext(maskRow);

        // Rotate the row vector from [0, D-1] to [i*D, (i+1)*D - 1] (shift is -i*D)
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
    std::cout << "FHE Matrix Multiplication & Attention Score Kernels Test (CPU Only)" << std::endl;

    const uint32_t L = 4; // Sequence length
    const uint32_t D = 8; // Head dimension

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(10); // Higher depth needed for Ct-Ct mults
    parameters.SetScalingModSize(50);
    parameters.SetFirstModSize(60);
    parameters.SetBatchSize(32); // At least L * D = 32

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    uint32_t M_actual = cc->GetCyclotomicOrder();
    uint32_t N_actual = cc->GetRingDimension();
    std::cout << "CryptoContext actual cyclotomic order M = " << M_actual 
              << ", actual ring dimension N = " << N_actual << std::endl;

    std::cout << "Generating keys..." << std::endl;
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalSumKeyGen(keyPair.secretKey);

    // Need all shifts for rotation
    std::vector<int32_t> rotationShifts;
    for (int32_t s = -32; s <= 32; ++s) {
        if (s != 0) rotationShifts.push_back(s);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotationShifts);

    std::mt19937 prng(12345);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    // Generate random Q, K, V
    std::vector<double> flatQ(L * D), flatK(L * D), flatV(L * D);
    for (uint32_t i = 0; i < L * D; ++i) {
        flatQ[i] = dist(prng);
        flatK[i] = dist(prng);
        flatV[i] = dist(prng);
    }

    // Encrypt
    auto ctQ = cc->Encrypt(keyPair.publicKey, cc->MakeCKKSPackedPlaintext(flatQ));
    auto ctK = cc->Encrypt(keyPair.publicKey, cc->MakeCKKSPackedPlaintext(flatK));
    auto ctV = cc->Encrypt(keyPair.publicKey, cc->MakeCKKSPackedPlaintext(flatV));

    // Test Ct-Ct Q x K^T
    std::cout << "\nEvaluating Q x K^T..." << std::endl;
    
    // Print debug info before calling EvalMatMulCtCtQTKT
    {
        uint32_t M = ctQ->GetCryptoParameters()->GetElementParams()->GetCyclotomicOrder();
        uint32_t autoIdx = cc->GetScheme()->FindAutomorphismIndex(-1, M);
        std::cout << "Debug QK^T: M = " << M << ", FindAutomorphismIndex(-1, M) = " << autoIdx << std::endl;
        
        auto keyTag = ctQ->GetKeyTag();
        const auto& keyMap = cc->GetEvalSumKeyMap(keyTag);
        std::cout << "Debug QK^T: keyMap size = " << keyMap.size() << ". Indices in map: ";
        for (const auto& pair : keyMap) {
            std::cout << pair.first << " ";
        }
        std::cout << std::endl;
    }

    auto ctScores = EvalMatMulCtCtQTKT(cc, ctQ, ctK, L, D);

    Plaintext ptScores;
    cc->Decrypt(keyPair.secretKey, ctScores, &ptScores);
    ptScores->SetLength(L * L);
    auto resScores = ptScores->GetCKKSPackedValue();

    // Check ground truth for Q x K^T
    double maxErrQK = 0.0;
    std::vector<double> expectedScores(L * L, 0.0);
    std::cout << "FHE Scores (QK^T):\n";
    for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t j = 0; j < L; ++j) {
            double truth = 0.0;
            for (uint32_t k = 0; k < D; ++k) {
                truth += flatQ[i * D + k] * flatK[j * D + k];
            }
            expectedScores[i * L + j] = truth;
            double fheVal = resScores[i * L + j].real();
            maxErrQK = std::max(maxErrQK, std::abs(fheVal - truth));
            std::cout << fheVal << " ";
        }
        std::cout << "\n";
    }
    std::cout << "Max QK^T Error: " << maxErrQK << std::endl;

    // Test Ct-Ct Scores x V
    std::cout << "\nEvaluating Scores x V..." << std::endl;
    auto ctOut = EvalMatMulCtCtScoresV(cc, ctScores, ctV, L, D);

    Plaintext ptOut;
    cc->Decrypt(keyPair.secretKey, ctOut, &ptOut);
    ptOut->SetLength(L * D);
    auto resOut = ptOut->GetCKKSPackedValue();

    double maxErrOut = 0.0;
    std::cout << "FHE Output vs Ground Truth (Scores x V):\n";
    for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t j = 0; j < D; ++j) {
            double truth = 0.0;
            for (uint32_t k = 0; k < L; ++k) {
                truth += expectedScores[i * L + k] * flatV[k * D + j];
            }
            double fheVal = resOut[i * D + j].real();
            maxErrOut = std::max(maxErrOut, std::abs(fheVal - truth));
            std::cout << "O[" << i << "][" << j << "]: FHE=" << fheVal << ", Truth=" << truth << ", Err=" << std::abs(fheVal - truth) << "\n";
        }
    }
    std::cout << "Max Scores x V Error: " << maxErrOut << std::endl;

    if (maxErrQK < 1e-5 && maxErrOut < 1e-5) {
        std::cout << "\nALL TESTS PASSED: FHE Attention Matrix Kernels evaluate correctly!" << std::endl;
    } else {
        std::cout << "\nFAILURE: Errors exceed tolerance." << std::endl;
    }

    return 0;
}
