mod sumtree;

use godot::prelude::*;
use rand::Rng;
use sumtree::{ReplayItem, SumTree};

const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;
const PRIORITY_ALPHA: f32 = 0.6;
const PRIORITY_BETA_START: f32 = 0.4;
const PRIORITY_BETA_STEPS: f32 = 100000.0;

struct DqnRustExtension;

#[gdextension]
unsafe impl ExtensionLibrary for DqnRustExtension {}

#[allow(non_snake_case)]
#[derive(GodotClass)]
#[class(base=Node, init)]
struct DQNRust {
    base: Base<Node>,

    state_dim: usize,
    action_dim: usize,
    hidden: usize,

    w1: Vec<f32>,
    b1: Vec<f32>,
    wA: Vec<f32>,
    bA: Vec<f32>,
    wV: Vec<f32>,
    bV: f32,

    w1t: Vec<f32>,
    b1t: Vec<f32>,
    wAt: Vec<f32>,
    bAt: Vec<f32>,
    wVt: Vec<f32>,
    bVt: f32,

    m_w1: Vec<f32>,
    v_w1: Vec<f32>,
    m_b1: Vec<f32>,
    v_b1: Vec<f32>,
    m_wA: Vec<f32>,
    v_wA: Vec<f32>,
    m_bA: Vec<f32>,
    v_bA: Vec<f32>,
    m_wV: Vec<f32>,
    v_wV: Vec<f32>,
    m_bV: f32,
    v_bV: f32,
    adam_step: i64,

    sumtree: SumTree,
    max_priority: f32,
    nstep_buffer: Vec<ReplayItem>,
    n_steps: usize,

    step_count: i64,
    epsilon: f64,
    gamma_pow: Vec<f32>,
}

#[godot_api]
impl DQNRust {
    #[func]
    fn init(&mut self, state_dim: i32, action_dim: i32, hidden: i32, replay_capacity: i32, n_steps: i32, gamma: f64) {
        let sd = state_dim as usize;
        let ad = action_dim as usize;
        let h = hidden as usize;

        self.state_dim = sd;
        self.action_dim = ad;
        self.hidden = h;
        self.n_steps = n_steps as usize;

        let mut gp = Vec::with_capacity(self.n_steps + 1);
        let mut p = 1.0_f32;
        gp.push(p);
        for _ in 0..self.n_steps {
            p *= gamma as f32;
            gp.push(p);
        }
        self.gamma_pow = gp;

        let mut rng = rand::thread_rng();
        let w1_len = h * sd;
        self.w1 = (0..w1_len).map(|_| rng.gen::<f32>() * 0.2 - 0.1).collect();
        self.b1 = vec![0.0f32; h];
        self.wA = (0..ad * h).map(|_| rng.gen::<f32>() * 0.2 - 0.1).collect();
        self.bA = vec![0.0f32; ad];
        self.wV = (0..h).map(|_| rng.gen::<f32>() * 0.2 - 0.1).collect();
        self.bV = 0.0;

        let zero_w1 = vec![0.0f32; w1_len];
        let zero_h = vec![0.0f32; h];
        let zero_ah = vec![0.0f32; ad * h];
        let zero_a = vec![0.0f32; ad];
        self.m_w1 = zero_w1.clone();
        self.v_w1 = zero_w1;
        self.m_b1 = zero_h.clone();
        self.v_b1 = zero_h.clone();
        self.m_wA = zero_ah.clone();
        self.v_wA = zero_ah;
        self.m_bA = zero_a.clone();
        self.v_bA = zero_a;
        self.m_wV = zero_h.clone();
        self.v_wV = zero_h;
        self.m_bV = 0.0;
        self.v_bV = 0.0;
        self.adam_step = 0;

        self.w1t = self.w1.clone();
        self.b1t = self.b1.clone();
        self.wAt = self.wA.clone();
        self.bAt = self.bA.clone();
        self.wVt = self.wV.clone();
        self.bVt = self.bV;

        self.sumtree = SumTree::new(replay_capacity as usize);
        self.max_priority = 1.0;
        self.nstep_buffer = Vec::with_capacity(self.n_steps);
        self.step_count = 0;
        self.epsilon = 1.0;
    }

    fn forward_inner(&self, x: &[f32], use_target: bool) -> (Vec<f32>, f32, Vec<f32>, Vec<f32>) {
        let (w1, b1, wA, bA, wV, bV) = if use_target {
            (&self.w1t, &self.b1t, &self.wAt, &self.bAt, &self.wVt, self.bVt)
        } else {
            (&self.w1, &self.b1, &self.wA, &self.bA, &self.wV, self.bV)
        };

        let sd = self.state_dim;
        let ad = self.action_dim;
        let h = self.hidden;

        let mut hidden = Vec::with_capacity(h);
        for hi in 0..h {
            let mut s = b1[hi];
            for j in 0..sd {
                s += w1[hi * sd + j] * x[j];
            }
            hidden.push(if s > 0.0 { s } else { 0.0 });
        }

        let mut V = bV;
        for hi in 0..h {
            V += wV[hi] * hidden[hi];
        }

        let mut A = Vec::with_capacity(ad);
        for ai in 0..ad {
            let mut s = bA[ai];
            for hi in 0..h {
                s += wA[ai * h + hi] * hidden[hi];
            }
            A.push(s);
        }

        let meanA = A.iter().sum::<f32>() / ad as f32;
        let Q: Vec<f32> = A.iter().map(|a| V + a - meanA).collect();

        (hidden, V, A, Q)
    }

    #[func]
    fn forward(&self, x: PackedFloat32Array, use_target: bool) -> Dictionary<GString, Variant> {
        let x_vec: Vec<f32> = x.to_vec();
        let (h, V, A, Q) = self.forward_inner(&x_vec, use_target);
        let mut dict: Dictionary<GString, Variant> = Dictionary::new();
        dict.set("h", &PackedFloat32Array::from(h.as_slice()));
        dict.set("V", &Variant::from(V as f64));
        dict.set("A", &PackedFloat32Array::from(A.as_slice()));
        dict.set("Q", &PackedFloat32Array::from(Q.as_slice()));
        dict
    }

    #[func]
    fn predict_q(&self, state: PackedFloat32Array) -> PackedFloat32Array {
        let x_vec: Vec<f32> = state.to_vec();
        let (_, _, _, Q) = self.forward_inner(&x_vec, false);
        PackedFloat32Array::from(Q.as_slice())
    }

    #[func]
    fn select_action(&mut self, state: PackedFloat32Array) -> i32 {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.epsilon {
            return rng.gen_range(0..self.action_dim) as i32;
        }
        let x_vec: Vec<f32> = state.to_vec();
        let (_, _, _, Q) = self.forward_inner(&x_vec, false);
        let mut best = 0;
        for i in 1..Q.len() {
            if Q[i] > Q[best] {
                best = i;
            }
        }
        best as i32
    }

    #[func]
    fn push_replay(&mut self, state: PackedFloat32Array, action: i32, reward: f64, next_state: PackedFloat32Array, done: bool) {
        let item = ReplayItem {
            state: state.to_vec(),
            action,
            reward: reward as f32,
            next_state: next_state.to_vec(),
            done,
            n_actual: 1,
        };
        self.nstep_buffer.push(item);

        if self.nstep_buffer.len() > self.n_steps {
            self.nstep_buffer.remove(0);
        }

        let push_ready = done || self.nstep_buffer.len() == self.n_steps;
        if !push_ready {
            return;
        }

        let mut G = 0.0f32;
        let mut final_idx = self.nstep_buffer.len() - 1;
        for i in 0..self.nstep_buffer.len() {
            G += self.gamma_pow[i] * self.nstep_buffer[i].reward;
            if self.nstep_buffer[i].done {
                final_idx = i;
                break;
            }
        }

        let first = &self.nstep_buffer[0];
        let last = &self.nstep_buffer[final_idx];
        let n_actual = (final_idx + 1) as i32;
        let p = self.max_priority.powf(PRIORITY_ALPHA);

        self.sumtree.add(ReplayItem {
            state: first.state.clone(),
            action: first.action,
            reward: G,
            next_state: last.next_state.clone(),
            done: last.done,
            n_actual,
        }, p);

        if done {
            self.nstep_buffer.clear();
        }
    }

    #[func]
    fn train(&mut self, batch_size: i32, _gamma: f64, grad_clip: f64, lr: f64) -> bool {
        let bs = batch_size as usize;
        if self.sumtree.size < bs {
            return false;
        }

        let (batch, indices, priorities) = self.sumtree.sample(bs);
        if batch.is_empty() {
            return false;
        }

        let total_p = self.sumtree.total();
        let n = self.sumtree.size as f32;
        let beta = (PRIORITY_BETA_START + self.step_count as f32 * (1.0 - PRIORITY_BETA_START) / PRIORITY_BETA_STEPS).min(1.0);

        let is_weights: Vec<f32> = (0..bs).map(|i| {
            let prob = priorities[i] / total_p;
            (1.0 / (n * prob + 1e-8)).powf(beta)
        }).collect();

        let sd = self.state_dim;
        let ad = self.action_dim;
        let h = self.hidden;

        let mut dw1 = vec![0.0f32; h * sd];
        let mut db1 = vec![0.0f32; h];
        let mut dwA = vec![0.0f32; ad * h];
        let mut dbA = vec![0.0f32; ad];
        let mut dwV = vec![0.0f32; h];
        let mut dbV = 0.0f32;

        for i in 0..bs {
            let item = &batch[i];
            let s = &item.state;
            let a = item.action as usize;
            let r = item.reward;
            let ns = &item.next_state;
            let d = item.done;
            let n_actual = item.n_actual as usize;

            let (h_vec, _V_s, _A_s, Q_s) = self.forward_inner(s, false);
            let (_, _, _, Q_on) = self.forward_inner(ns, false);

            let mut best_a = 0;
            for j in 1..Q_on.len() {
                if Q_on[j] > Q_on[best_a] {
                    best_a = j;
                }
            }

            let (_, _, _, Q_tg) = self.forward_inner(ns, true);
            let target_val = if d { 0.0 } else { self.gamma_pow[n_actual].min(1.0) * Q_tg[best_a] };
            let target = r + target_val;

            let td_err = (Q_s[a] - target).abs() + 1e-6;
            let p = td_err.powf(PRIORITY_ALPHA);
            self.sumtree.set_priority(indices[i], p);
            if td_err > self.max_priority {
                self.max_priority = td_err;
            }

            let mut dQ = 2.0 * (Q_s[a] - target) / bs as f32;
            dQ *= is_weights[i];

            let dV_grad = dQ;
            let mut dA_grad = vec![-dQ / ad as f32; ad];
            dA_grad[a] += dQ;

            for hi in 0..h {
                dwV[hi] += dV_grad * h_vec[hi];
            }
            dbV += dV_grad;

            for ai in 0..ad {
                let dai = dA_grad[ai];
                for hi in 0..h {
                    dwA[ai * h + hi] += dai * h_vec[hi];
                }
                dbA[ai] += dai;
            }

            let mut grad_h = vec![0.0f32; h];
            for hi in 0..h {
                let mut ssum = dV_grad * self.wV[hi];
                for ai in 0..ad {
                    ssum += dA_grad[ai] * self.wA[ai * h + hi];
                }
                grad_h[hi] = if h_vec[hi] > 0.0 { ssum } else { 0.0 };
            }

            for hi in 0..h {
                let ghi = grad_h[hi];
                for j in 0..sd {
                    dw1[hi * sd + j] += ghi * s[j];
                }
                db1[hi] += ghi;
            }
        }

        let gc = grad_clip as f32;
        if gc > 0.0 {
            for v in dw1.iter_mut() { *v = v.clamp(-gc, gc); }
            for v in db1.iter_mut() { *v = v.clamp(-gc, gc); }
            for v in dwA.iter_mut() { *v = v.clamp(-gc, gc); }
            for v in dbA.iter_mut() { *v = v.clamp(-gc, gc); }
            for v in dwV.iter_mut() { *v = v.clamp(-gc, gc); }
            dbV = dbV.clamp(-gc, gc);
        }

        self.step_count += 1;
        self.adam_step += 1;
        let current_lr = lr as f32;
        let b1c = 1.0 - ADAM_BETA1.powi(self.adam_step as i32);
        let b2c = 1.0 - ADAM_BETA2.powi(self.adam_step as i32);

        apply_adam(&dw1, &mut self.w1, &mut self.m_w1, &mut self.v_w1, current_lr, b1c, b2c);
        apply_adam(&db1, &mut self.b1, &mut self.m_b1, &mut self.v_b1, current_lr, b1c, b2c);
        apply_adam(&dwA, &mut self.wA, &mut self.m_wA, &mut self.v_wA, current_lr, b1c, b2c);
        apply_adam(&dbA, &mut self.bA, &mut self.m_bA, &mut self.v_bA, current_lr, b1c, b2c);
        apply_adam(&dwV, &mut self.wV, &mut self.m_wV, &mut self.v_wV, current_lr, b1c, b2c);

        self.m_bV = ADAM_BETA1 * self.m_bV + (1.0 - ADAM_BETA1) * dbV;
        self.v_bV = ADAM_BETA2 * self.v_bV + (1.0 - ADAM_BETA2) * dbV * dbV;
        let mbV_hat = self.m_bV / b1c;
        let vbV_hat = self.v_bV / b2c;
        self.bV -= current_lr * mbV_hat / (vbV_hat.sqrt() + ADAM_EPS);

        let tau = 0.005f32;
        polyak(&self.w1, &mut self.w1t, tau);
        polyak(&self.b1, &mut self.b1t, tau);
        polyak(&self.wA, &mut self.wAt, tau);
        polyak(&self.bA, &mut self.bAt, tau);
        polyak(&self.wV, &mut self.wVt, tau);
        self.bVt = tau * self.bV + (1.0 - tau) * self.bVt;

        true
    }

    #[func]
    fn get_epsilon(&self) -> f64 { self.epsilon }

    #[func]
    fn set_epsilon(&mut self, eps: f64) { self.epsilon = eps; }

    #[func]
    fn get_step_count(&self) -> i64 { self.step_count }

    #[func]
    fn set_step_count(&mut self, count: i64) { self.step_count = count; }

    #[func]
    fn get_adam_step(&self) -> i64 { self.adam_step }

    #[func]
    fn set_adam_step(&mut self, step: i64) { self.adam_step = step; }

    #[func]
    fn get_replay_size(&self) -> i64 { self.sumtree.size as i64 }

    #[func]
    fn get_max_priority(&self) -> f64 { self.max_priority as f64 }

    #[func]
    fn set_max_priority(&mut self, p: f64) { self.max_priority = p as f32; }

    #[func]
    fn get_weights_online(&self) -> Array<Variant> {
        let mut arr: Array<Variant> = Array::new();
        arr.push(&PackedFloat32Array::from(self.w1.as_slice()));
        arr.push(&PackedFloat32Array::from(self.b1.as_slice()));
        arr.push(&PackedFloat32Array::from(self.wA.as_slice()));
        arr.push(&PackedFloat32Array::from(self.bA.as_slice()));
        arr.push(&PackedFloat32Array::from(self.wV.as_slice()));
        arr.push(&Variant::from(self.bV as f64));
        arr
    }

    #[func]
    fn set_weights_online(&mut self, weights: Array<Variant>) {
        if weights.len() >= 6 {
            if let Some(v) = weights.get(0) { if let Ok(pf) = v.try_to::<PackedFloat32Array>() { self.w1 = pf.to_vec(); } }
            if let Some(v) = weights.get(1) { if let Ok(pf) = v.try_to::<PackedFloat32Array>() { self.b1 = pf.to_vec(); } }
            if let Some(v) = weights.get(2) { if let Ok(pf) = v.try_to::<PackedFloat32Array>() { self.wA = pf.to_vec(); } }
            if let Some(v) = weights.get(3) { if let Ok(pf) = v.try_to::<PackedFloat32Array>() { self.bA = pf.to_vec(); } }
            if let Some(v) = weights.get(4) { if let Ok(pf) = v.try_to::<PackedFloat32Array>() { self.wV = pf.to_vec(); } }
            if let Some(v) = weights.get(5) { if let Ok(bv) = v.try_to::<f64>() { self.bV = bv as f32; } }
        }
    }

    #[func]
    fn get_adam_state(&self) -> Array<Variant> {
        let mut arr: Array<Variant> = Array::new();
        arr.push(&PackedFloat32Array::from(self.m_w1.as_slice()));
        arr.push(&PackedFloat32Array::from(self.v_w1.as_slice()));
        arr.push(&PackedFloat32Array::from(self.m_b1.as_slice()));
        arr.push(&PackedFloat32Array::from(self.v_b1.as_slice()));
        arr.push(&PackedFloat32Array::from(self.m_wA.as_slice()));
        arr.push(&PackedFloat32Array::from(self.v_wA.as_slice()));
        arr.push(&PackedFloat32Array::from(self.m_bA.as_slice()));
        arr.push(&PackedFloat32Array::from(self.v_bA.as_slice()));
        arr.push(&PackedFloat32Array::from(self.m_wV.as_slice()));
        arr.push(&PackedFloat32Array::from(self.v_wV.as_slice()));
        arr.push(&Variant::from(self.m_bV as f64));
        arr.push(&Variant::from(self.v_bV as f64));
        arr.push(&Variant::from(self.adam_step as i64));
        arr
    }

    #[func]
    fn set_adam_state(&mut self, state: Array<Variant>) {
        if state.len() >= 13 {
            let mut idx = 0;
            if let Some(v) = state.get(idx) { if let Ok(pf) = v.try_to::<PackedFloat32Array>() { self.m_w1 = pf.to_vec(); } } idx += 1;
            if let Some(v) = state.get(idx) { if let Ok(pf) = v.try_to::<PackedFloat32Array>() { self.v_w1 = pf.to_vec(); } } idx += 1;
            if let Some(v) = state.get(idx) { if let Ok(pf) = v.try_to::<PackedFloat32Array>() { self.m_b1 = pf.to_vec(); } } idx += 1;
            if let Some(v) = state.get(idx) { if let Ok(pf) = v.try_to::<PackedFloat32Array>() { self.v_b1 = pf.to_vec(); } } idx += 1;
            if let Some(v) = state.get(idx) { if let Ok(pf) = v.try_to::<PackedFloat32Array>() { self.m_wA = pf.to_vec(); } } idx += 1;
            if let Some(v) = state.get(idx) { if let Ok(pf) = v.try_to::<PackedFloat32Array>() { self.v_wA = pf.to_vec(); } } idx += 1;
            if let Some(v) = state.get(idx) { if let Ok(pf) = v.try_to::<PackedFloat32Array>() { self.m_bA = pf.to_vec(); } } idx += 1;
            if let Some(v) = state.get(idx) { if let Ok(pf) = v.try_to::<PackedFloat32Array>() { self.v_bA = pf.to_vec(); } } idx += 1;
            if let Some(v) = state.get(idx) { if let Ok(pf) = v.try_to::<PackedFloat32Array>() { self.m_wV = pf.to_vec(); } } idx += 1;
            if let Some(v) = state.get(idx) { if let Ok(pf) = v.try_to::<PackedFloat32Array>() { self.v_wV = pf.to_vec(); } } idx += 1;
            if let Some(v) = state.get(idx) { if let Ok(bv) = v.try_to::<f64>() { self.m_bV = bv as f32; } } idx += 1;
            if let Some(v) = state.get(idx) { if let Ok(bv) = v.try_to::<f64>() { self.v_bV = bv as f32; } } idx += 1;
            if let Some(v) = state.get(idx) { if let Ok(sv) = v.try_to::<i64>() { self.adam_step = sv; } }
        }
    }

    #[func]
    fn copy_to_target(&mut self) {
        self.w1t = self.w1.clone();
        self.b1t = self.b1.clone();
        self.wAt = self.wA.clone();
        self.bAt = self.bA.clone();
        self.wVt = self.wV.clone();
        self.bVt = self.bV;
    }

    #[func]
    fn polyak(&mut self, tau: f64) {
        let t = tau as f32;
        polyak(&self.w1, &mut self.w1t, t);
        polyak(&self.b1, &mut self.b1t, t);
        polyak(&self.wA, &mut self.wAt, t);
        polyak(&self.bA, &mut self.bAt, t);
        polyak(&self.wV, &mut self.wVt, t);
        self.bVt = t * self.bV + (1.0 - t) * self.bVt;
    }

    #[func]
    fn _ready(&mut self) {
        godot_print!("DQNRust agent ready");
    }
}

fn apply_adam(dw: &[f32], w: &mut [f32], m: &mut [f32], v: &mut [f32], lr: f32, b1c: f32, b2c: f32) {
    for i in 0..w.len() {
        let g = dw[i];
        m[i] = ADAM_BETA1 * m[i] + (1.0 - ADAM_BETA1) * g;
        v[i] = ADAM_BETA2 * v[i] + (1.0 - ADAM_BETA2) * g * g;
        let m_hat = m[i] / b1c;
        let v_hat = v[i] / b2c;
        w[i] -= lr * m_hat / (v_hat.sqrt() + ADAM_EPS);
    }
}

fn polyak(src: &[f32], dst: &mut [f32], tau: f32) {
    for i in 0..src.len() {
        dst[i] = tau * src[i] + (1.0 - tau) * dst[i];
    }
}
