use rand::Rng;

#[derive(Clone, Debug)]
pub struct ReplayItem {
    pub state: Vec<f32>,
    pub action: i32,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
    pub n_actual: i32,
}

pub struct SumTree {
    tree: Vec<f32>,
    data: Vec<Option<ReplayItem>>,
    capacity: usize,
    pub size: usize,
    write_idx: usize,
}

impl Default for SumTree {
    fn default() -> Self {
        Self {
            tree: vec![0.0f32; 2],
            data: vec![None],
            capacity: 1,
            size: 0,
            write_idx: 0,
        }
    }
}

impl SumTree {
    pub fn new(cap: usize) -> Self {
        let mut capacity = 1;
        while capacity < cap {
            capacity <<= 1;
        }
        let tree = vec![0.0f32; capacity * 2];
        let data = vec![None; capacity];
        Self {
            tree,
            data,
            capacity,
            size: 0,
            write_idx: 0,
        }
    }

    pub fn total(&self) -> f32 {
        self.tree[1]
    }

    pub fn add(&mut self, item: ReplayItem, priority: f32) {
        let idx = self.capacity + self.write_idx;
        self.data[self.write_idx] = Some(item);
        self._tree_set(idx, priority);
        self.write_idx = (self.write_idx + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
    }

    fn _tree_set(&mut self, mut idx: usize, priority: f32) {
        self.tree[idx] = priority;
        idx >>= 1;
        while idx > 0 {
            self.tree[idx] = self.tree[idx * 2] + self.tree[idx * 2 + 1];
            idx >>= 1;
        }
    }

    pub fn retrieve(&self, mut idx: usize, mut s: f32) -> usize {
        while idx < self.capacity {
            let left = idx * 2;
            if self.tree[left] >= s {
                idx = left;
            } else {
                s -= self.tree[left];
                idx = left + 1;
            }
        }
        idx
    }

    pub fn sample(&self, n: usize) -> (Vec<ReplayItem>, Vec<usize>, Vec<f32>) {
        let total_p = self.total();
        if total_p <= 0.0 {
            return (vec![], vec![], vec![]);
        }
        let mut rng = rand::thread_rng();
        let seg = total_p / n as f32;
        let mut batch = Vec::with_capacity(n);
        let mut indices = Vec::with_capacity(n);
        let mut priorities = Vec::with_capacity(n);
        for i in 0..n {
            let s = seg * (i as f32 + rng.gen::<f32>());
            let idx = self.retrieve(1, s);
            let data_idx = idx - self.capacity;
            indices.push(data_idx);
            if let Some(ref item) = self.data[data_idx] {
                batch.push(item.clone());
            }
            priorities.push(self.tree[idx]);
        }
        (batch, indices, priorities)
    }

    pub fn set_priority(&mut self, idx: usize, priority: f32) {
        if idx < self.capacity {
            self._tree_set(self.capacity + idx, priority);
        }
    }
}
