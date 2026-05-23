#include "openfhe.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace lbcrypto;

// Helper to evaluate accelerated sign
Ciphertext<DCRTPoly> Eval_accelerated_sign(const CryptoContext<DCRTPoly>& cc, Ciphertext<DCRTPoly> x, int df, int dg) {
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

Ciphertext<DCRTPoly> Eval_accelerated_comparison(const CryptoContext<DCRTPoly>& cc, Ciphertext<DCRTPoly> x, double threshold, int df, int dg) {
    auto diff = cc->EvalSub(x, threshold);
    // Scale down to [-1, 1] domain to prevent polynomial divergence
    // Maximum input difference is ~ 14 (10 - (-4)), so we scale by 1/16
    diff = cc->EvalMult(diff, 1.0/16.0);
    
    auto sign_val = Eval_accelerated_sign(cc, diff, df, dg);
    
    auto res = cc->EvalAdd(sign_val, 1.0);
    return cc->EvalMult(res, 0.5);
}

Ciphertext<DCRTPoly> approx_gelu_piecewise(const CryptoContext<DCRTPoly>& cc, Ciphertext<DCRTPoly> x, int df=2, int dg=2) {
    auto c_neg4 = Eval_accelerated_comparison(cc, x, -4.0, df, dg);
    auto c_neg1_95 = Eval_accelerated_comparison(cc, x, -1.95, df, dg);
    auto c_3 = Eval_accelerated_comparison(cc, x, 3.0, df, dg);
    
    // ind_0 = 1 - c_neg4
    auto ind_0 = cc->EvalSub(1.0, c_neg4);
    
    // ind_1 = c_neg4 * (1 - c_neg1_95)
    auto ind_1_sub = cc->EvalSub(1.0, c_neg1_95);
    auto ind_1 = cc->EvalMult(c_neg4, ind_1_sub);
    
    // ind_2 = c_neg1_95 * (1 - c_3)
    auto ind_2_sub = cc->EvalSub(1.0, c_3);
    auto ind_2 = cc->EvalMult(c_neg1_95, ind_2_sub);
    
    // ind_3 = c_3
    auto ind_3 = c_3;
    
    std::vector<double> coeff_f0 = {-0.5054031199708174, -0.42226581151983866, -0.11807612951181953, -0.011034134030615728};
    std::vector<double> coeff_f1 = {0.008526321541038084, 0.5, 0.3603292692789629, 0.0, -0.037688200365904236, 0.0, 0.0018067462606141187};
    
    // Evaluate branches
    // val_0 is just 0
    auto val_1 = cc->EvalPoly(x, coeff_f0);
    auto val_2 = cc->EvalPoly(x, coeff_f1);
    auto val_3 = x; // Just x
    
    // Multiplex: ind_1 * val_1 + ind_2 * val_2 + ind_3 * val_3
    auto branch1 = cc->EvalMult(ind_1, val_1);
    auto branch2 = cc->EvalMult(ind_2, val_2);
    auto branch3 = cc->EvalMult(ind_3, val_3);
    
    auto result = cc->EvalAdd(branch1, branch2);
    result = cc->EvalAdd(result, branch3);
    
    return result;
}

int main() {
    std::cout << "GELU Piecewise Approximation Test (CPU Only)" << std::endl;

    CCParams<CryptoContextCKKSRNS> parameters;
    
    // With df=2, dg=2, each sign eval is depth 16.
    // +2 for indicators, +3 for f1, +1 for multiplexing.
    // Let's use depth 22.
    parameters.SetMultiplicativeDepth(22);
    parameters.SetScalingModSize(50);
    parameters.SetFirstModSize(60);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    std::vector<std::complex<double>> input;
    std::vector<double> expected_true_gelu;
    
    int num_values = 32;
    for (int i = 0; i < num_values; ++i) {
        double val = -10.0 + (20.0 * i) / (num_values - 1); 
        input.push_back(val);
        
        double ground_truth_gelu = 0.5 * val * (1.0 + std::erf(val / std::sqrt(2.0)));
        expected_true_gelu.push_back(ground_truth_gelu);
    }
    
    Plaintext plaintext = cc->MakeCKKSPackedPlaintext(input);

    std::cout << "Generating keys (this will take a while due to massive depth)..." << std::endl;
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);

    std::cout << "Encrypting..." << std::endl;
    auto ciphertext = cc->Encrypt(keyPair.publicKey, plaintext);

    std::cout << "Evaluating Piecewise GELU..." << std::endl;
    auto result_ct = approx_gelu_piecewise(cc, ciphertext, 2, 2);

    std::cout << "Decrypting..." << std::endl;
    Plaintext plaintext_res;
    cc->Decrypt(keyPair.secretKey, result_ct, &plaintext_res);
    plaintext_res->SetLength(input.size());

    double max_err = 0.0;
    std::vector<std::complex<double>> res_vals = plaintext_res->GetCKKSPackedValue();

    for (size_t i = 0; i < input.size(); ++i) {
        double out_val = res_vals[i].real();
        double truth = expected_true_gelu[i];
        
        max_err = std::max(max_err, std::abs(out_val - truth));
        std::cout << "Input: " << input[i].real() << " | FHE: " << out_val << " | Truth: " << truth << std::endl;
    }

    std::cout << "Max Absolute Error (FHE Piecewise vs True GELU): " << max_err << std::endl;

    if (max_err < 0.1) {
        std::cout << "SUCCESS: The piecewise polynomial evaluates extreme values accurately!" << std::endl;
    } else {
        std::cout << "WARNING: Max error is larger than expected. The scaling factor of 1/16 in the sign comparison might have blunted the sharp transitions." << std::endl;
    }

    return 0;
}
