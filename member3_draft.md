# 3. Rust DQN GDExtension Implementation

## 3.1 Overview

We implement a full Dueling DQN with Prioritized Experience Replay, N-step returns, Double DQN, and the Adam optimizer—all in Rust, compiled as a Godot 4 GDExtension. The `DQNRust` class is compiled to `libdqn_rust.so` (~4.8 MB) and loaded by the engine at startup via the GDExtension loader config (`dqn_rust.gdextension`), which specifies the entry symbol `gdext_rust_init` and the platform library path. GDScript creates the agent with `DQNRust.new()` and calls `agent.init(...)`, `agent.select_action(...)`, and `agent.train(...)`—all heavy computation runs in native code [Example1_Simple.gd:109-110]. No external ML libraries, no GPU—just safe Rust with `rand` 0.8 and the `godot` 0.5 crate bindings [Cargo.toml].

## 3.2 Network Architecture

The agent uses two hidden layers: 512 → 256 units, each with ReLU activation. This is an upgrade from the original GDScript version, which used only a single 256-hidden-layer network. Weight matrices are stored as flat `Vec<f32>` in row-major order (`w1`: 512 × state_dim, `w2`: 256 × 512, `wA`: 7 × 256, `wV`: 256) for cache-friendly linear iteration [lib.rs:32-49]. From the second hidden layer, two streams emerge:

- **Value stream**: Hidden(256) → V (scalar)
- **Advantage stream**: Hidden(256) → A (7 action values)

The output is computed via the Dueling aggregation (Wang et al. 2016):

\[
Q(s, a) = V(s) + A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')
\]

This architecture decouples state value from action advantage, which is particularly suited to flight control where simply maintaining altitude is valuable regardless of the specific control input. The target network (`w1t, b1t, ..., wVt, bVt`) is an identical copy maintained via Polyak soft updates. See `rust_dqn/src/lib.rs` lines 166-199 for the `forward_inner` function.

## 3.3 State Space (12 or 13 dimensions)

*Same as the GDScript version—the state space is defined by the GDScript environment, not the Rust agent.*

| Index | Variable | Normalization |
|-------|----------|---------------|
| 0 | forward_air_speed | / 100 |
| 1 | local_altitude | / 500 |
| 2 | sin(pitch) | — |
| 3 | cos(pitch) | — |
| 4 | sin(roll) | — |
| 5 | cos(roll) | — |
| 6 | vertical_speed | / 50 |
| 7 | is_stalled | 0 or 1 |
| 8-10 | obstacle distances (3 angles) | / max_dist |
| 11 | fuel SoC | 0-1 |
| 12 (Complex) | battery SoC | 0-1 |

All variables are normalized to roughly \([-1, 1]\) or \([0, 1]\) via the shown transforms. The Simple scenario (Example1) uses 12 dimensions; the Complex scenario (Example2) adds battery SoC for a total of 13.

## 3.4 Action Space (7 discrete)

*Same as GDScript version—defined by the environment.*

| Action | Effect |
|--------|--------|
| 0 | nothing |
| 1 | pitch up (+0.2) |
| 2 | pitch down (−0.2) |
| 3 | roll left (−0.2) |
| 4 | roll right (+0.2) |
| 5 | throttle up (+0.1) |
| 6 | throttle down (−0.1) |

The agent is agnostic to action semantics—it receives `PackedFloat32Array` states and returns integer action indices via `select_action()` [lib.rs:222-236].

## 3.5 Forward Pass (Rust)

The forward pass computes a two-layer ReLU network using flat-vector dot products. The core helper function `relu_activations(x, w, b, n_out, n_in)` [lib.rs:154-164] is shared between both hidden layers:

```rust
fn relu_activations(x: &[f32], w: &[f32], b: &[f32], n_out: usize, n_in: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let mut s = b[i];
        for j in 0..n_in {
            s += w[i * n_in + j] * x[j];
        }
        out.push(if s > 0.0 { s } else { 0.0 });
    }
    out
}
```

The inner loop—`for j in 0..n_in { s += w[i * n_in + j] * x[j] }`—is the performance-critical kernel. Unlike GDScript, which interprets each iteration with bytecode dispatch overhead, Rust's tight loop compiles to a short, predictable machine-code sequence that LLVM auto-vectorizes into SIMD instructions. The `forward_inner()` function [lib.rs:166-199] calls `relu_activations` for both layers, then computes the value and advantage streams from the resulting 256-unit hidden representation, finally aggregating them into 7 Q-values via the Dueling equation.

## 3.6 Backpropagation & Training (Rust)

The `train()` method [lib.rs:289-457] performs manual gradient computation through both hidden layers. The gradient flow is:

\[
\frac{\partial L}{\partial Q} \;\to\; \frac{\partial L}{\partial V} + \frac{\partial L}{\partial A} \;\to\; \frac{\partial L}{\partial h_2} \;\to\; \frac{\partial L}{\partial h_1} \;\to\; \frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial b_1}
\]

For each sample in the batch of 64:

1. **Forward pass** through online and target networks to obtain \(Q(s, \cdot)\) and \(Q_{\text{target}}(s', \cdot)\)
2. **Double DQN**: the online network picks the best action in the next state; the target network evaluates it: \(y = r + \gamma^n Q_{\text{target}}(s', \arg\max_a Q_{\text{online}}(s', a))\)
3. **TD error** \(\delta = |Q_{\text{online}}(s, a) - y|\) triggers a priority update in the SumTree
4. **Gradient accumulation** into all parameter groups: `dw1, db1, dw2, db2, dwA, dbA, dwV, dbV`

Gradients are accumulated across all 64 samples with mean normalization, then clipped element-wise to \([-1.0, 1.0]\) [lib.rs:414-424].

## 3.7 Why Rust Eliminates Chunking

This is the single biggest practical improvement of the Rust port. The GDScript version split batch-64 into 4 chunks of 16 because each chunk took ~8 ms in the interpreted loop. At 60 FPS the frame budget is ~16 ms—one chunk was feasible, but all 64 at once caused ~35 ms spikes that dropped frames. Rust's native loop processes all 64 samples in well under 1 ms because LLVM auto-vectorizes the inner dot-product loops. The result is no chunking, no frame budget management, and simpler GDScript code.

## 3.8 Prioritized Experience Replay

The replay buffer is a SumTree data structure (binary heap tree) providing O(log n) priority updates, implemented in `rust_dqn/src/sumtree.rs` [sumtree.rs:33-113]. The tree is stored as a flat `Vec<f32>` of size \(2 \times \text{capacity}\) (internal nodes hold subtree sums; leaves hold transition priorities) alongside a `Vec<Option<ReplayItem>>` data array of capacity-sized slots.

Key parameters:
- \(\alpha = 0.6\) controls prioritization strength
- \(\beta\) anneals linearly from 0.4 → 1.0 over 100,000 steps to gradually reduce importance sampling bias
- New transitions are inserted with the current `max_priority` to ensure they are sampled at least once
- Importance sampling weights \(w_i = (N \cdot P(i))^{-\beta}\) correct the prioritization bias in gradient updates [lib.rs:304-307]

N-step returns (n = 3) are pre-processed in `push_replay()` [lib.rs:239-286], computing discounted cumulative reward \(G_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k}\) before inserting into the SumTree.

## 3.9 Adam Optimizer (Rust)

The Adam optimizer is a full implementation with first and second moment estimates and bias correction. It is applied via the `apply_adam(dw, w, m, v, lr, b1c, b2c)` helper [lib.rs:594-603], which applies a single loop over the parameter vector:

```rust
fn apply_adam(dw: &[f32], w: &mut [f32], m: &mut [f32], v: &mut [f32],
              lr: f32, b1c: f32, b2c: f32) {
    for i in 0..w.len() {
        let g = dw[i];
        m[i] = ADAM_BETA1 * m[i] + (1.0 - ADAM_BETA1) * g;
        v[i] = ADAM_BETA2 * v[i] + (1.0 - ADAM_BETA2) * g * g;
        let m_hat = m[i] / b1c;
        let v_hat = v[i] / b2c;
        w[i] -= lr * m_hat / (v_hat.sqrt() + ADAM_EPS);
    }
}
```

The optimizer is called for all 8 parameter groups (`w1, b1, w2, b2, wA, bA, wV, bV`) each training step [lib.rs:432-444]. The scalar bias `bV` is handled separately since it is a single float. After the Adam update, Polyak soft-target updates (\(\tau = 0.005\)) blend the online weights into the target network [lib.rs:446-454].

## 3.10 Comparison: GDScript vs. Rust

| Aspect | GDScript | Rust (GDExtension) |
|--------|----------|-------------------|
| Hidden layers | 1 × 256 | 2 × (512 → 256) |
| Batch-64 training time | ~35 ms (chunked to 4 × ~8 ms) | ≤1 ms (single pass) |
| Replay capacity | 50,000 | 100,000 |
| Lines of DQN code | ~500 per file | ~510 total |
| Training throughput | Limited by frame budget | No frame budget concern |

The Rust port achieves a 35× reduction in training time while supporting a larger network (2× the hidden units), a larger replay buffer (2× capacity), and simpler GDScript code with no chunking logic.

## Key Source Files

- `rust_dqn/src/lib.rs` (609 lines) — entire DQN agent: forward pass, backpropagation, Adam optimizer, serialization, and godot-rust bindings
- `rust_dqn/src/sumtree.rs` (114 lines) — prioritized experience replay SumTree with O(log n) insert, sample, and priority update
- `rust_dqn/Cargo.toml` — manifest with `cdylib` target, `godot` 0.5 and `rand` 0.8 dependencies
- `dqn_rust.gdextension` — Godot 4 extension loader configuration (entry symbol, minimum version, platform library path)
- `example/Example1_Simple.gd` lines 62, 109-110 — GDScript-side agent creation (`DQNRust.new()`) and initialization (`agent.init(...)`)
- `docs/dqn_rust.md` — full API reference and architecture documentation
