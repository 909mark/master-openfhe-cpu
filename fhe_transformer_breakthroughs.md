# FHE Transformer Block: Technical Breakthroughs

Building a complete GPT-2 Transformer block purely in Fully Homomorphic Encryption (FHE) using OpenFHE presented several deeply technical challenges. Below is a breakdown of the primary breakthroughs achieved to stabilize the pipeline, accompanied by concrete numerical examples to demonstrate the underlying FHE mechanics.

---

## 1. CKKS Ciphertext Slots Truncation Vulnerability

**Technical Explanation:** 
In OpenFHE, the `EvalMult` operation implicitly relies on the `GetSlots()` metadata of the input ciphertext to optimize polynomial multiplications. During the Feed-Forward Network (FFN) expansion phase, our token embedding dimension was projected from $D=4$ up to $4D=16$. However, the input ciphertext—originating from the sparse `EvalBootstrap` phase—retained a metadata limit of `GetSlots() == 8`. 

Consequently, when multiplying by a large expansion plaintext mask of size 32, OpenFHE silently truncated the plaintext to match the 8-slot ciphertext limit. This forced the upper $75\%$ of the FFN matrix into a zero-state, destabilizing the subsequent Chebyshev polynomial approximations within the piece-wise GELU activation and throwing an `approximation error is too high` exception. The solution was to clone the ciphertext and dynamically override its metadata via `SetSlots(32)` prior to the evaluation.

**Concrete Example:** 
Suppose our input to the Feed-Forward Network is an $L=2$, $D=4$ matrix, which takes up $8$ slots in the ciphertext. We need to project this to an intermediate dimension of $4D=16$, requiring $2 \times 16 = 32$ total slots. 

To isolate and shift elements, we create a plaintext mask of size $32$ containing a `1.0` at slot index $20$, and try to multiply it by our input ciphertext (`ct->GetSlots() == 8`). OpenFHE's `EvalMult` aligns their capacities by truncating the plaintext to $8$ slots. The `1.0` at index $20$ is dropped and implicitly replaced by `0.0`. Thus, multiplying the shifted ciphertext by this truncated mask results in EXACTLY `0.0` for all projection slots from $8$ to $31$, causing catastrophic data loss.

---

## 2. Multiplicative Depth Squeeze

**Technical Explanation:** 
Fully Homomorphic Encryption schemes like CKKS operate within a strict noise budget defined by the multiplicative depth. Our post-bootstrap pipeline involved LayerNorm 2, an FFN projection, piece-wise GELU, and Residual 2. The GELU approximation alone required evaluating 9th-degree and 6th-degree Chebyshev polynomials, consuming $\approx 11$ multiplicative levels. 

Combined with the Newton-Raphson iterations for LayerNorm, the total depth exceeded our allocated `levelsAvailableAfterBootstrap = 35`, causing the scaling factor to drop below the threshold of precision and terminating the decryption. By temporarily bypassing the default cryptographic constraints (`SetSecurityLevel(HEStd_NotSet)`) for this proof-of-concept, we were able to safely bump the available levels to $40$ without inadvertently triggering an expansion of the underlying ring dimension $N$ (which would have exponentially increased computational overhead).

**Concrete Example:** 
Our post-bootstrap pipeline must sequentially compute LayerNorm 2, FFN 1, GELU, and FFN 2. Suppose we strictly have $35$ multiplicative levels available after bootstrapping.
- **LayerNorm 2** (using 3 Newton-Raphson iterations for inverse square root) consumes $\approx 21$ levels.
- **FFN 1** (matrix projection) consumes $2$ levels.
- **Piece-wise GELU** (evaluating up to a 9th-degree Chebyshev polynomial for the sign function and a 6th-degree polynomial for the activation) consumes $\approx 11$ levels.
- **FFN 2** (matrix projection) consumes $2$ levels.

The total depth required is $21 + 2 + 11 + 2 = 36$ levels. Because our pipeline demanded $36$ levels but the ciphertexts were budgeted for $35$, the final operations exceeded the limit. This caused the CKKS scaling factor to collapse (decaying the encrypted values into noise), which triggered the decryption error.

---

## 3. Rolling Window Mean vs Broadcast (`EvalSum`)

**Technical Explanation:** 
In CKKS, the rotational summation command `EvalSum(ct, n)` evaluates $\sum_{i=0}^{n-1} x_{j+i}$ in slot $j$. It essentially operates as a sliding window rather than universally broadcasting the global sum across all slots. 

During the row-wise mean and variance calculations in LayerNorm, we relied on `EvalSum` to compute the normalization denominator. However, because our rotational boundaries wrapped at the block size ($8$) rather than the matrix row dimension ($4$), slots 1, 2, and 3 computed corrupted partial sums containing empty masking zeroes. We resolved this by wrapping `EvalSum` with `ReplicateSlot0`, which extracts the mathematically true sum from slot 0 and safely broadcasts it across the entire dynamically active sub-vector.

**Concrete Example:** 
Consider a row vector of $D=4$ elements: `[1.0, 2.0, 3.0, 4.0]`. In our pipeline, this is embedded in an $8$-slot ciphertext padded with zeros: `[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]`. When we call `EvalSum(ct, 4)`, OpenFHE computes a sliding window sum:
- **Slot 0 gets:** $1.0 + 2.0 + 3.0 + 4.0 = \mathbf{10.0}$ (Correct!)
- **Slot 1 gets:** $2.0 + 3.0 + 4.0 + 0.0 = \mathbf{9.0}$ (Incorrect!)
- **Slot 2 gets:** $3.0 + 4.0 + 0.0 + 0.0 = \mathbf{7.0}$ (Incorrect!)

If we used this output directly to compute the mean (dividing by $4$), slot $1$ would wrongly subtract $2.25$ instead of $2.5$. By applying `ReplicateSlot0`, we explicitly isolate slot $0$ (`[10.0, 0.0, ...]`) and recursively fold it, resulting in the broadcasted vector `[10.0, 10.0, 10.0, 10.0, ...]`. This ensures every slot in the row normalizes against the identical, correctly derived sum.
