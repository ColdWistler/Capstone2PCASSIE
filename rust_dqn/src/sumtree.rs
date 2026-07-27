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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_item(id: i32) -> ReplayItem {
        ReplayItem {
            state: vec![id as f32; 3],
            action: 0,
            reward: 0.0,
            next_state: vec![0.0; 3],
            done: false,
            n_actual: 1,
        }
    }

    #[test]
    fn ut01_add_and_total() {
        let mut tree = SumTree::new(8);
        assert_eq!(tree.total(), 0.0);
        assert_eq!(tree.size, 0);

        tree.add(make_item(0), 1.0);
        assert_eq!(tree.size, 1);
        assert!((tree.total() - 1.0).abs() < 1e-6);

        tree.add(make_item(1), 2.0);
        assert_eq!(tree.size, 2);
        assert!((tree.total() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn ut01_update_priority() {
        let mut tree = SumTree::new(4);
        tree.add(make_item(0), 1.0);
        tree.add(make_item(1), 2.0);
        assert!((tree.total() - 3.0).abs() < 1e-6);

        tree.set_priority(0, 5.0);
        assert!((tree.total() - 7.0).abs() < 1e-6);

        let item = tree.data[0].as_ref().unwrap();
        assert_eq!(item.state[0], 0.0);
    }

    #[test]
    fn ut02_sample_proportional_to_priority() {
        let mut tree = SumTree::new(8);
        tree.add(make_item(0), 1.0);
        tree.add(make_item(1), 3.0);
        tree.add(make_item(2), 0.0);
        tree.add(make_item(3), 1.0);

        let total = tree.total();
        assert!((total - 5.0).abs() < 1e-6);

        let mut counts = [0u32; 4];
        for _ in 0..1000 {
            let (batch, _, _) = tree.sample(1);
            if !batch.is_empty() {
                let id = batch[0].state[0] as usize;
                counts[id] += 1;
            }
        }

        assert!(counts[1] > counts[0], "higher priority item sampled more often");
        assert!(counts[0] > counts[2], "zero priority item sampled least");
        assert_eq!(counts[2], 0, "zero priority never sampled");
    }

    #[test]
    fn ut03_min_priority_tracking() {
        let mut tree = SumTree::new(4);
        tree.add(make_item(0), 10.0);
        tree.add(make_item(1), 0.5);
        tree.add(make_item(2), 5.0);

        let (batch, _indices, priorities) = tree.sample(3);
        assert_eq!(batch.len(), 3);

        let min_p = priorities.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_p = priorities.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(min_p <= max_p);
    }

    #[test]
    fn ut02_sample_returns_correct_batch_size() {
        let mut tree = SumTree::new(8);
        for i in 0..5 {
            tree.add(make_item(i), 1.0);
        }
        let (batch, indices, priorities) = tree.sample(3);
        assert_eq!(batch.len(), 3);
        assert_eq!(indices.len(), 3);
        assert_eq!(priorities.len(), 3);
    }

    #[test]
    fn ut01_wrapping_write_index() {
        let mut tree = SumTree::new(4);
        for i in 0..8 {
            tree.add(make_item(i), 1.0);
        }
        assert_eq!(tree.size, 4);
        assert!((tree.total() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn ut03_empty_tree_sample() {
        let tree = SumTree::new(4);
        let (batch, indices, priorities) = tree.sample(3);
        assert!(batch.is_empty());
        assert!(indices.is_empty());
        assert!(priorities.is_empty());
    }
}
