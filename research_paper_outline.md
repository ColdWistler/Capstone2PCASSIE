# Research Paper: Capstone2PCASSIE — Rust-Accelerated DRL for Autonomous Flight

**Combined angles:** (1) Deep RL implemented in Rust as a Godot GDExtension, (2) GPU-free native performance via godot-rust bindings, (3) Transfer learning across aircraft configurations, (4) Real-time training pipeline engineering

**Proposed title:** *"Rust-Accelerated Deep Reinforcement Learning for Autonomous Flight Control: A GDExtension Dueling DQN in Godot 4"*

---

## How to Use This Document

1. Read your assigned part below
2. Read the source files listed — they contain the code and context you need
3. If something is unclear, you can ask the AI assistant directly with a prompt like:
   - *"Explain the Dueling DQN architecture in rust_dqn/src/lib.rs lines 150-200"*
   - *"Explain how the Rust backpropagation works in the train() function"*
   - *"Explain the modular aircraft architecture in addons/simplified_flightsim/"*
   - *"What are the state space and action space for the DQN agent?"*
   - *"Explain the transfer learning approach between Example1 and Example2"*
4. Write your section, then we merge and edit together

---

## Member 1 — Introduction & Related Work

### What to write (~4 pages)

**1.1 Introduction**
- Problem: Training RL agents in game engines typically requires external ML libraries (PyTorch via GDNative, TensorFlow via C++ bindings). Alternatively, implementing the full DQN in a scripting language like GDScript is simple but suffers from poor training throughput (interpreted loops bottleneck the physics frame).
- We present a **hybrid approach**: the simulation environment runs in GDScript, but the neural network forward pass, backpropagation, and optimizer are implemented as a **Rust GDExtension** (`DQNRust` class) compiled to native code.
- The system controls aircraft in a modular 3D flight simulator within the Godot 4.7 engine.
- We study two contributions: (a) the engineering of a Rust-via-GDExtension DRL training pipeline that eliminates interpreted-loop bottlenecks, (b) whether policies transfer between aircraft of differing complexity.
- Brief overview of the system (flight sim → Rust DQN agent → training pipeline → transfer experiment)

**1.2 Related Work**
- Deep RL in games: DQN in Atari (Mnih et al.), RL in game engines
- Godot ML approaches: Godot RL agents (C#/Python), GDNative ML integrations — contrast with our Rust GDExtension approach using godot-rust bindings
- Flight simulation for RL: MS Flight Sim + ML, X-Plane + RL, AirSim — contrast with our simplified physics
- DQN variants: Dueling DQN (Wang et al.), Prioritized Experience Replay (Schaul et al.), Double DQN (van Hasselt et al.), N-step returns (Sutton)
- Transfer learning in RL: transferring policies across environment variants
- **Rust for game engine ML**: godot-rust (the `godot` crate), GDExtension vs. GDNative — our work is among the first to use Rust GDExtension for real-time DRL training

**1.3 Problem Statement & Contributions**
- Formally state the 3 contributions: (1) first Rust GDExtension DQN with full training pipeline inside Godot 4, (2) elimination of interpreted-loop bottleneck enabling larger networks (512→256 vs. previous 256) and larger replay buffers (100k vs. 50k) without frame drops, (3) transfer learning study across aircraft configurations

### Key source files
- `README.md` (project overview)
- `docs/dqn_rust.md` (technical reference — architecture, API, performance)
- `docs/CHANGELOG.md` (history of changes from GDScript to Rust)
- `rust_dqn/src/lib.rs` lines 1-50 (constants, struct definition, init)

---

## Member 2 — Flight Physics Engine Architecture

### What to write (~4 pages)

*This section is largely unchanged from the GDScript version — the physics engine is still pure GDScript and was not ported to Rust.*

**2.1 Overview of Simplified FlightSim**
- Godot 4 plugin providing modular flight physics
- Aircraft is a `RigidBody3D` with swappable child modules
- Designed for games (not real-world accuracy), but physically plausible

**2.2 Core Physics Model**
- Lift equation: `L = Cl * rho * V² / 2 * A`
- Drag equation: 3-axis drag (DragFactor vector), DragPoint
- Gravity, stall detection, g-force calculation
- Atmospheric model: air density + temperature vary with altitude
- Linear (flat) vs. Spherical (planet) world modes

**2.3 Modular Architecture**
- `AircraftModule` base class (`base_module.gd`) — defines `setup()`, `receive_input()`, `process_physic_frame()`, `process_render_frame()` lifecycle
- Module discovery via `ModuleType` string tags
- Each module: `Engine` (thrust, fuel, sound), `Steering` (ailerons/elevator/rudder), `Flaps` (lift/drag modifiers), `LandingGear` (deploy/stow/collision), `EnergyContainer` (fuel/battery), `InstrumentAttitude` (roll/pitch/heading)
- Control modules: `ControlSteering`, `ControlEngine`, `ControlFlaps`, `ControlLandingGear`

**2.4 How Modularity Enables Multi-Aircraft Configs**
- Simple (Example1): 1 engine, 1 fuel tank, no temperature
- Complex (Example2): 4 engines, 2 fuel tanks + battery, temperature, mountains
- Helicopter (Example3): vertical-lift engines, no wings
- Space (Example4): spherical world, multiple planets, vacuum
- Same architecture, different configurations — this enables transfer learning

### Key source files
- `addons/simplified_flightsim/Aircraft/Aircraft.gd`
- `addons/simplified_flightsim/aircraft_modules/base_module.gd`
- Example scripts state extraction (e.g. `Example1_Simple.gd` lines 502-516)

---

## Member 3 — Rust DQN Implementation (GDExtension)

### What to write (~5 pages)

**3.1 Overview**
- A full Dueling DQN with prioritized replay, N-step returns, Double DQN, and Adam optimizer — all implemented in Rust as a Godot 4 GDExtension
- The `DQNRust` class is compiled to `libdqn_rust.so` (~4.8 MB) and loaded by the engine at startup
- GDScript creates the agent with `DQNRust.new()` and calls `agent.init(...)` / `agent.select_action(...)` / `agent.train(...)` — the heavy computation runs in native code
- No external ML libraries, no GPU — just safe Rust with `rand` and the `godot` crate bindings

**3.2 Network Architecture**
- Two hidden layers: 512 → 256, ReLU activations (upgraded from single 256-hidden layer in the GDScript version)
- Weight matrices stored as flat `Vec<f32>` (row-major) for cache-friendly iteration
- Two streams from the second hidden layer:
  - Value stream: Hidden(256) → V (scalar)
  - Advantage stream: Hidden(256) → A (7 action values)
- Output: `Q(s,a) = V(s) + A(s,a) - mean(A(s,:))`
- This is the Dueling architecture (Wang et al. 2016)
- See `rust_dqn/src/lib.rs` lines 150-200 (forward_inner function)

**3.3 State Space (12 or 13 dimensions)**

*Same as the GDScript version — the state space is defined by the GDScript environment, not the Rust agent.*

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

**3.4 Action Space (7 discrete)**

*Same as GDScript version — defined by the environment.*

| Action | Effect |
|--------|--------|
| 0 | nothing |
| 1 | pitch up (+0.2) |
| 2 | pitch down (-0.2) |
| 3 | roll left (-0.2) |
| 4 | roll right (+0.2) |
| 5 | throttle up (+0.1) |
| 6 | throttle down (-0.1) |

**3.5 Forward Pass (Rust)**
- Two-layer ReLU computation with flat-vector dot products
- Inner loop is a tight `for` loop over weights — compiled to SIMD-friendly LLVM IR
- Unlike GDScript (which interprets each iteration), Rust's `for j in 0..n_in { s += w[i * n_in + j] * x[j] }` compiles to a short, predictable machine-code loop
- Helper function `relu_activations(x, w, b, n_out, n_in)` is shared between both layers
- See `rust_dqn/src/lib.rs` lines 157-168

**3.6 Backpropagation & Training (Rust)**
- Manual gradient computation through both hidden layers
- Gradient flow: `dQ → dV + dA → grad_h2 → grad_h1 → dw1,db1`
- For each sample in the batch:
  1. Forward pass through online and target networks
  2. Double DQN: online picks best action, target evaluates it
  3. TD error → priority update for SumTree
  4. Gradient accumulation into `dw1, db1, dw2, db2, dwA, dbA, dwV, dbV`
- Gradient clipping to [-1.0, 1.0]
- See `rust_dqn/src/lib.rs` lines 296-380

**3.7 Why Rust Eliminates Chunking**
- The GDScript version split batch-64 into 4 chunks of 16 because each chunk took ~8ms in the interpreted loop (at 60 FPS, a frame budget is ~16ms — one chunk was feasible, but all 64 at once caused ~35ms spikes)
- Rust's native loop processes all 64 samples in well under 1ms (LLVM auto-vectorizes the inner dot-product loops)
- Result: no chunking, no frame budget management, simpler GDScript code
- This is the single biggest practical improvement of the Rust port

**3.8 Prioritized Experience Replay**
- SumTree data structure (binary heap tree) for O(log n) priority updates
- Stored in `rust_dqn/src/sumtree.rs` as a flat `Vec<f32>` tree + `Vec<Option<ReplayItem>>` data array
- `alpha = 0.6` for prioritization, `beta` anneals from 0.4 → 1.0 over 100k steps
- Importance sampling weights correct the prioritization bias

**3.9 Adam Optimizer (Rust)**
- Full Adam implementation: first/second moment estimates, bias correction
- Applied via `apply_adam(dw, w, m, v, lr, b1c, b2c)` helper — one loop over the parameter vector
- Called for all 7 parameter groups (w1,b1,w2,b2,wA,bA,wV,bV) each train step
- Polyak soft target updates (tau = 0.005) applied after every training step

**3.10 Comparison: GDScript vs. Rust**
| Aspect | GDScript | Rust (GDExtension) |
|---|---|---|
| Hidden layers | 1 × 256 | 2 × (512 → 256) |
| Batch-64 training time | ~35 ms (chunked to 4×8ms) | ≤1 ms (single pass) |
| Replay capacity | 50,000 | 100,000 |
| Lines of DQN code | ~500 per file | ~510 total |
| Training throughput | Limited by frame budget | No frame budget concern |

### Key source files
- `rust_dqn/src/lib.rs` (entire file — 527 lines, the full Rust DQN)
- `rust_dqn/src/sumtree.rs` (114 lines — prioritized replay SumTree)
- `rust_dqn/Cargo.toml` (dependencies)
- `dqn_rust.gdextension` (extension loader config)
- `example/Example1_Simple.gd` lines 61-62, 105-106 (agent creation and init — the GDScript side of the bridge)
- `docs/dqn_rust.md` (API reference)

---

## Member 4 — Real-Time Training Pipeline & Performance

### What to write (~4 pages)

**4.1 The Core Challenge**
- Training neural networks is computationally expensive
- Godot runs a real-time game loop — any lag spike causes visible stutter
- GDScript is interpreted, ~10-50x slower than equivalent native code
- Our solution: offload the entire DQN computation to a **Rust GDExtension** compiled to native machine code

**4.2 Frame Budgeting (Simplified)**
- `FRAMES_PER_STEP = 4`: only 1 DQN action per 4 physics frames (15 actions/sec)
- `TRAIN_INTERVAL = 2`: training runs every 2 actions, not every action
- Between actions, physics still runs — the plane doesn't freeze
- **Key difference from GDScript version**: No chunking needed. The Rust `train()` call processes all 64 samples in one shot and returns well within the frame budget

**4.3 Headless Mode**
- `--headless` flag detected via `OS.has_feature("headless")`
- Disables rendering, uncaps FPS (`Engine.max_fps = -1`), sets 5× time scale
- Physics, raycasts, and game logic still run at full speed
- Result: ~6.7 seconds per 2000-step episode vs ~33 seconds in rendered mode
- Any student can reproduce this with just `godot --headless --path .`

**4.4 Training Throughput Analysis**
- Measurements:
  - Rendered: ~33s / episode
  - Headless 5×: ~6.7s / episode
  - Rust training: consistent frame pacing, no spikes at any batch size
  - GDScript training (for reference): visible stutter without chunking, manageable with chunking
- The Rust agent introduces negligible per-frame overhead (<0.5ms for full train pass)

**4.5 Save/Load & Multi-Run Infrastructure**
- Auto-saves every 50 episodes
- Multi-run tracking with auto-incrementing run IDs
- Best model auto-selected across runs on startup
- Adam state preserved in save format (weights array of 8 elements + Adam state of 17 elements)
- SAVE_VERSION = 5 (incompatible with old GDScript format)

**4.6 Python Tooling for Analysis**
- `telemetry_dashboard.py`: Streamlit real-time dashboard
- `resource_monitor.py`: PyQt6 CPU/RAM/GPU/disk monitor with auto-launch (optional)

### Key source files
- `example/Example1_Simple.gd` — `_physics_process` (frame budgeting, agent calls)
- `rust_dqn/src/lib.rs` — `train()` function (single-pass batch processing)
- `docs/CHANGELOG.md` — before/after performance comparison
- `build_rust_dqn.sh` — rebuild script

---

## Member 5 — Experiments, Transfer Learning & Results

### What to write (~5 pages)

This member needs to **run the actual code** to collect results. Do this first, then write.

**5.1 Experimental Setup**
- Hardware: note CPU, RAM, GPU used
- Software: Godot 4.7, Rust 1.96, godot-rust 0.5, headless mode
- Hyperparameters (table): all constants from Example scripts
- Two scenarios: Simple (Example1) and Complex (Example2)

**5.2 Experiment 1: Simple Scenario Training**
- Train from scratch for N episodes (e.g., 2000+)
- Report: reward curve (episode vs. reward), epsilon decay, best reward, landing success rate
- Use `--test` mode after training to get quantitative eval over 10 episodes
- Include: avg reward, avg steps, landing%, crash%, avg altitude, avg stalls
- Generate the HTML report and include key charts

**5.3 Experiment 2: Complex Scenario Training**
- Train from scratch on Example2_Complex
- Same metrics as Experiment 1
- Compare convergence speed to Simple (expect slower due to harder problem)

**5.4 Experiment 3: Transfer Learning (Simple → Complex)**
- Take trained Simple model weights
- Architecture: Simple has STATE_DIM=12, Complex has STATE_DIM=13 (extra battery SoC input)
- Approach: load Simple weights, zero-initialize the 13th column of w1 (the new weight connections for battery)
- Continue training on Complex scenario
- Compare convergence speed vs. training Complex from scratch
- Hypothesis: pre-training on Simple should accelerate learning on Complex (shared flight dynamics, only new element is battery management)

**5.5 Results Comparison**
- Table comparing all 3 experiments
- Key metrics: episodes to converge, best reward, landing success rate, avg altitude deviation, stall frequency
- Reward curves overlaid on same plot (Simple, Complex from scratch, Complex with transfer)

**5.6 Discussion**
- Why did transfer help (or not)? Analysis of shared vs. distinct state dimensions
- Impact of Rust acceleration on experiment throughput (can run more episodes in less time → better hyperparameter tuning)
- Limitations: small state space, simplified physics, discrete actions, scripted landing bypasses DQN
- Generalizability of the Rust GDExtension approach to other Godot projects

**5.7 Conclusion**
- Summarize findings
- Key takeaway: Rust GDExtension DRL is feasible and offers significant practical advantages over pure-scripting implementations (no frame budget management, larger networks, simpler code)
- Future work: continuous action space, PPO, multi-agent, real hardware, GPU acceleration via Rust bindings (wgpu, CUBLAS)

### How to run experiments

```bash
# Build the Rust extension first
./build_rust_dqn.sh

# Train Simple (headless, let it run for a while)
godot --headless --path .

# Test Simple
godot --headless --path . -- --test

# Train Complex
godot --headless --path . res://example/Example2_Complex.tscn

# Test Complex
godot --headless --path . res://example/Example2_Complex.tscn -- --test
```

Save the training CSV logs and test reports for charts.

### Key source files
- `example/Example1_Simple.gd` (Simple training loop, reward, state, actions)
- `example/Example2_Complex.gd` (Complex variant)
- `rust_dqn/src/lib.rs` (Rust agent core)
- `docs/CHANGELOG.md` (reward shaping details)

---

## Paper Structure (Final Assembly Order)

1. **Title + Abstract** — (all, but Member 5 drafts based on results)
2. **Introduction** — Member 1
3. **Related Work** — Member 1
4. **Flight Physics Architecture** — Member 2
5. **Rust DQN GDExtension Implementation** — Member 3
6. **Real-Time Training Pipeline** — Member 4
7. **Experimental Setup** — Member 5
8. **Results** — Member 5
9. **Discussion** — Member 5 (with input from all)
10. **Conclusion** — Member 5 (with input from all)

Each section should be ~3-5 pages for a total of ~20-25 pages. Co-author all sections, but each member leads their assigned part.
