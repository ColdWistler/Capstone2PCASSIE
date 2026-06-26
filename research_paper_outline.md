# Research Paper: Capstone2PCASSIE — In-Engine DRL for Autonomous Flight

**Combined angles:** (1) Deep RL from scratch in GDScript, (2) Transfer learning across aircraft configurations, (4) Real-time training pipeline engineering

**Proposed title:** *"In-Engine Deep Reinforcement Learning for Autonomous Flight Control: A Pure GDScript Dueling DQN in Godot 4"*

---

## How to Use This Document

1. Read your assigned part below
2. Read the source files listed — they contain the code and context you need
3. If something is unclear, you can ask the AI assistant directly with a prompt like:
   - *"Explain the Dueling DQN architecture in Example1_Simple.gd lines 825-856"*
   - *"Explain how the chunked training works in lines 924-1063 of Example1_Simple.gd"*
   - *"Explain the modular aircraft architecture in addons/simplified_flightsim/"*
   - *"What are the state space and action space for the DQN agent?"*
   - *"Explain the transfer learning approach between Example1 and Example2"*
4. Write your section, then we merge and edit together

---

## Member 1 — Introduction & Related Work

### What to write (~4 pages)

**1.1 Introduction**
- Problem: Training RL agents in game engines typically requires external ML libraries (PyTorch via GDNative, TensorFlow via C++ bindings). We present a complete Dueling DQN *implemented entirely in GDScript* within the Godot 4 engine.
- The system controls aircraft in a modular 3D flight simulator
- We study two contributions: (a) the engineering of real-time NN training inside a game loop without external deps, (b) whether policies transfer between aircraft of differing complexity
- Brief overview of the system (flight sim → DQN agent → training pipeline → transfer experiment)

**1.2 Related Work**
- Deep RL in games: DQN in Atari (Mnih et al.), RL in game engines
- Godot ML approaches: Godot RL agents (C#/Python), GDNative ML integrations — contrast with our pure GDScript approach
- Flight simulation for RL: MS Flight Sim + ML, X-Plane + RL, AirSim — contrast with our simplified physics
- DQN variants: Dueling DQN (Wang et al.), Prioritized Experience Replay (Schaul et al.), Double DQN (van Hasselt et al.), N-step returns (Sutton)
- Transfer learning in RL: transferring policies across environment variants

**1.3 Problem Statement & Contributions**
- Formally state the 3 contributions: (1) first pure-GDScript DQN with full training pipeline, (2) real-time training with chunked batched processing, (3) transfer learning study across aircraft configurations

### Key source files
- `README.md`, `COMMANDS.txt` (project overview)
- `Example1_Simple.gd` lines 1-36 (hyperparameters, architecture constants)
- `CHANGELOG.txt` (history of optimizations — useful for related work context)

---

## Member 2 — Flight Physics Engine Architecture

### What to write (~4 pages)

**2.1 Overview of Simplified FlightSim**
- Godot 4 plugin providing modular flight physics
- Aircraft is a `RigidBody3D` with swappable child modules
- Designed for games (not real-world accuracy), but physically plausible

**2.2 Core Physics Model**
- Lift equation: `L = Cl * rho * V² / 2 * A` (lines 51-52 of Aircraft.gd)
- Drag equation: 3-axis drag (DragFactor vector), DragPoint
- Gravity, stall detection, g-force calculation
- Atmospheric model: air density + temperature vary with altitude
- Linear (flat) vs. Spherical (planet) world modes

**2.3 Modular Architecture**
- `AircraftModule` base class (`base_module.gd`) — defines `setup()`, `receive_input()`, `process_physic_frame()`, `process_render_frame()` lifecycle
- Module discovery via `ModuleType` string tags (e.g. "engine", "steering")
- Each module: `Engine` (thrust, fuel, sound), `Steering` (ailerons/elevator/rudder forces), `Flaps` (lift/drag modifiers), `LandingGear` (deploy/stow/collision), `EnergyContainer` (fuel/battery), `InstrumentAttitude` (roll/pitch/heading)
- Control modules: `ControlSteering`, `ControlEngine`, `ControlFlaps`, `ControlLandingGear` — map keyboard to module inputs

**2.4 How Modularity Enables Multi-Aircraft Configs**
- Simple (Example1): 1 engine, 1 fuel tank, no temperature
- Complex (Example2): 4 engines, 2 fuel tanks + battery, temperature, mountains
- Helicopter (Example3): vertical-lift engines, no wings
- Space (Example4): spherical world, multiple planets, vacuum
- Same architecture, different configurations — this is what makes transfer learning possible

### Key source files
- `addons/simplified_flightsim/Aircraft/Aircraft.gd` (505 lines — core physics)
- `addons/simplified_flightsim/aircraft_modules/base_module.gd` (module lifecycle)
- `addons/simplified_flightsim/aircraft_modules/base_module_spatial.gd`
- `addons/simplified_flightsim/aircraft_modules/Engine/Engine.gd`
- `addons/simplified_flightsim/aircraft_modules/Steering/Steering.gd`
- `addons/simplified_flightsim/aircraft_modules/Flaps/Flaps.gd`
- `addons/simplified_flightsim/aircraft_modules/LandingGear/LandingGear.gd`
- `addons/simplified_flightsim/aircraft_modules/Instruments/InstrumentAttitude.gd`
- `addons/simplified_flightsim/aircraft_modules/EnergyContainer/EnergyContainer.gd`
- `addons/simplified_flightsim/aircraft_modules/Controls/ControlSteering.gd`
- `Example1_Simple.gd` lines 729-741 (state extraction from modules)
- `Example2_Complex.gd` lines 191-206 (module discovery for multi-engine)

---

## Member 3 — DQN Implementation (From Scratch in GDScript)

### What to write (~5 pages)

**3.1 Overview**
- A full Dueling DQN with prioritized replay, N-step returns, Double DQN, and Adam optimizer — all hand-implemented in ~1100 lines of GDScript
- No external ML libraries, no neural network framework — just arrays and math

**3.2 Network Architecture**
- Input: 12-D state vector (Simple) or 13-D (Complex)
- Hidden layer: 256 neurons, ReLU activation
- Two streams:
  - Value stream: Hidden(256) → V (scalar)
  - Advantage stream: Hidden(256) → A (7 action values)
- Output: `Q(s,a) = V(s) + A(s,a) - mean(A(s,:))`
- This is the Dueling architecture (Wang et al. 2016)
- See `Example1_Simple.gd` lines 825-856 (forward function)

**3.3 State Space (12 dimensions)**
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
| 8-10 | obstacle distances (fwd, left 15°, right 15°) | / max_dist |
| 11 | fuel SoC | 0-1 |

Complex adds dimension 12: battery SoC

**3.4 Action Space (7 discrete)**
| Action | Effect |
|--------|--------|
| 0 | nothing |
| 1 | pitch up (+0.2) |
| 2 | pitch down (-0.2) |
| 3 | roll left (-0.2) |
| 4 | roll right (+0.2) |
| 5 | throttle up (+0.1) |
| 6 | throttle down (-0.1) |

**3.5 Backpropagation & Training**
- Manual gradient computation: dQ/dV, dQ/dA, backprop through ReLU
- Gradient clipping to [-1.0, 1.0]
- See lines 985-1018 for gradient computation
- This is the most technically novel part — very few examples of hand-written backprop in GDScript exist

**3.6 Prioritized Experience Replay**
- SumTree data structure (binary heap tree) for O(log n) priority updates
- `alpha = 0.6` for prioritization, `beta` anneals from 0.4 → 1.0 over 100k steps
- Importance sampling weights correct the bias
- See lines 129-191 (SumTree class) and 903-921 (sampling)

**3.7 N-Step Returns**
- N=3: accumulates up to 3 steps of reward before bootstrapping
- Faster reward propagation, reduced bias
- See lines 876-901

**3.8 Double DQN**
- Online network selects best action, target network evaluates it
- Reduces Q-value overestimation
- See lines 968-977

**3.9 Adam Optimizer**
- Full Adam implementation: first/second moment estimates, bias correction
- LR decay: 0.001 * 0.9995^step, floor at 0.0001
- Polyak soft target updates (tau = 0.005) every step
- See lines 1036-1078

**3.10 Comparison with PyTorch Equivalent**
- Include a table showing lines of code / complexity comparison
- E.g., PyTorch's 10-line `nn.Linear` + `optim.Adam` vs. our ~100 lines of manual gradient + Adam

### Key source files
- `Example1_Simple.gd` lines 69-1078 (the entire DQN)
  - 69-120: weight variables, hyperparameters
  - 129-191: SumTree class
  - 244-291: network init, Adam init, target copy
  - 825-856: forward pass (Dueling architecture)
  - 859-873: predict/select action
  - 876-921: N-step push, prioritized sampling
  - 924-1063: train_step with chunking, gradient computation, Adam update, Polyak
  - 1066-1078: Adam helper, Polyak helper
- `Example2_Complex.gd` (same architecture, STATE_DIM=13)
- `CHANGELOG.txt` (Adam, N-step, chunking were added incrementally)

---

## Member 4 — Real-Time Training Pipeline & Performance

### What to write (~4 pages)

**4.1 The Core Challenge**
- Training neural networks is computationally expensive
- Godot runs a real-time game loop — any lag spike causes visible stutter
- GDScript is interpreted, ~10-50x slower than equivalent C++/PyTorch
- We need to run forward passes, backward passes, replay sampling, and Adam updates *within individual physics frames*

**4.2 Frame Budgeting**
- `FRAMES_PER_STEP = 4`: only 1 DQN action per 4 physics frames (15 actions/sec)
- `TRAIN_INTERVAL = 2`: training runs every 2 actions, not every action
- Between actions, physics still runs — the plane doesn't freeze

**4.3 Chunked Training (Key Innovation)**
- Problem: processing a full batch of 64 in one frame caused ~100ms+ spikes
- Solution: split batch of 64 into 4 chunks of 16, process one chunk per training call
- Gradients accumulate across chunks in `chunk_dw1`, `chunk_db1`, etc.
- Adam update + Polyak only applied after all chunks complete
- Per-frame work stays constant (~16 samples) regardless of total batch size
- See lines 109-120 (state vars), 924-1063 (chunked train_step logic)
- This was the critical fix that made training usable (CHANGELOG "Issue 4")

**4.4 Headless Mode**
- `--headless` flag detected via `OS.has_feature("headless")`
- Disables rendering, uncaps FPS (`Engine.max_fps = -1`), sets 5× time scale
- Physics, raycasts, and game logic still run at full speed
- Result: ~6.7 seconds per 2000-step episode vs ~33 seconds in rendered mode
- Any student can reproduce this with just `godot --headless --path .`

**4.5 Training Throughput Analysis**
- Measurements (from CHANGELOG / COMMANDS.txt):
  - Rendered: ~33s / episode
  - Headless 5×: ~6.7s / episode
  - With chunking: consistent frame pacing, no spikes
  - Without chunking: visible stutter every training frame
- Can include a table comparing throughput at different batch sizes, chunk sizes, frame skip values

**4.6 Save/Load & Multi-Run Infrastructure**
- Auto-saves every 50 episodes
- Multi-run tracking with auto-incrementing run IDs
- Best model auto-selected across runs on startup
- Adam state preserved in save format v4

**4.7 Python Tooling for Analysis**
- `telemetry_dashboard.py`: Streamlit real-time dashboard
- `generate_report.py`: static HTML report with Plotly charts (altitude, speed, attitude, g-force, etc.)
- `resource_monitor.py`: PyQt6 CPU/RAM/GPU/disk monitor with auto-launch
- These are secondary but show the full experimental ecosystem

### Key source files
- `Example1_Simple.gd`:
  - 109-120 (chunked training state variables)
  - 209-212 (headless mode detection)
  - 1081-1227 (`_physics_process` — main loop with frame budgeting)
- `CHANGELOG.txt` — especially:
  - "Headless training mode" (lines 25-37)
  - "Issue 2: Lag after training" (lines 88-95)
  - "Issue 3: Progressive lag" (lines 98-108)
  - "Issue 4: Lag spikes" (lines 110-126)
- `COMMANDS.txt` (run instructions)
- `resource_monitor.py`
- `telemetry_dashboard.py`
- `generate_report.py`
- `requirements-telemetry.txt`

---

## Member 5 — Experiments, Transfer Learning & Results

### What to write (~5 pages)

This member needs to **run the actual code** to collect results. Do this first, then write.

**5.1 Experimental Setup**
- Hardware: note CPU, RAM, GPU used
- Software: Godot 4.7, headless mode
- Hyperparameters (table): all constants from lines 3-36 of Example1_Simple.gd
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
- Architecture is identical except STATE_DIM: 12 → 13 (extra battery SoC input)
- Approach: load Simple weights, zero-initialize the 13th column of w1 (i.e., `w1[hi * STATE_DIM + 12] = 0` for all hi)
- Continue training on Complex scenario
- Compare convergence speed vs. training Complex from scratch
- Hypothesis: pre-training on Simple should accelerate learning on Complex (shared flight dynamics, only new element is battery management)

**5.5 Results Comparison**
- Table comparing all 3 experiments
- Key metrics: episodes to converge, best reward, landing success rate, avg altitude deviation, stall frequency
- Reward curves overlaid on same plot (Simple, Complex from scratch, Complex with transfer)

**5.6 Discussion**
- Why did transfer help (or not)? Analysis of shared vs. distinct state dimensions
- Limitations: small state space, simplified physics, discrete actions, scripted landing bypasses DQN
- Generalizability of the approach to other Godot projects

**5.7 Conclusion**
- Summarize findings
- Key takeaway: pure-GDScript DRL is feasible for simplified control tasks; transfer learning across aircraft configs shows promise with minimal architecture change
- Future work: continuous action space, PPO, multi-agent, real hardware

### How to run experiments

```bash
# Train Simple (headless, let it run for a while)
godot --headless --path /home/soggy/Documents/GitHub/Capstone2PCASSIE

# Test Simple
godot --headless --path /home/soggy/Documents/GitHub/Capstone2PCASSIE -- --test

# Train Complex
godot --headless --path /home/soggy/Documents/GitHub/Capstone2PCASSIE res://example/Example2_Complex.tscn

# Test Complex
godot --headless --path /home/soggy/Documents/GitHub/Capstone2PCASSIE res://example/Example2_Complex.tscn -- --test

# Generate HTML report (after test)
python3 generate_report.py
```

Save the training CSV logs and test reports for charts.

### Key source files
- `Example1_Simple.gd` (Simple training loop, reward, state, actions)
- `Example2_Complex.gd` (Complex variant, differences highlighted)
- `generate_report.py` (report generation)
- `CHANGELOG.txt` (reward shaping details — important for understanding what the agent optimizes)

---

## Paper Structure (Final Assembly Order)

1. **Title + Abstract** — (all, but Member 5 drafts based on results)
2. **Introduction** — Member 1
3. **Related Work** — Member 1
4. **Flight Physics Architecture** — Member 2
5. **DQN Implementation** — Member 3
6. **Real-Time Training Pipeline** — Member 4
7. **Experimental Setup** — Member 5
8. **Results** — Member 5
9. **Discussion** — Member 5 (with input from all)
10. **Conclusion** — Member 5 (with input from all)

Each section should be ~3-5 pages for a total of ~20-25 pages. Co-author all sections, but each member leads their assigned part.
