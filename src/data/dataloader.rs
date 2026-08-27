use super::dataset::Dataset;
use super::structures::Sample;
use super::utils;
use rand::prelude::SliceRandom;
use std::cmp::min;

pub struct DataLoader {
    dataset: Dataset,
    bs: usize,
    shuffle: bool,
    indices: Vec<usize>,
    curr_index: usize,
}

impl DataLoader {
    pub fn new(dataset: Dataset, bs: usize, shuffle: bool) -> Self {
        let curr_index: usize = 0;
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if shuffle {
            let mut rng = rand::rng();
            indices.shuffle(&mut rng);
        }
        Self {
            dataset,
            bs,
            shuffle,
            indices,
            curr_index,
        }
    }

    pub fn num_batches(&self) -> usize {
        self.dataset.len().div_ceil(self.bs)
    }

    pub fn reset(&mut self) {
        self.curr_index = 0;
        if self.shuffle {
            let mut rng = rand::rng();
            self.indices.shuffle(&mut rng);
        }
    }

    fn get_batch(&self, batch_idx: usize) -> Vec<Sample> {
        let start_idx = self.bs * batch_idx;
        let end_idx = min(start_idx + self.bs, self.dataset.len());
        let mut batch: Vec<Sample> = Vec::new();
        for idx in start_idx..end_idx {
            let real_idx = self.indices[idx];
            let path = self.dataset.get(real_idx);
            match utils::load_sample(path) {
                Ok(sample) => batch.push(sample),
                Err(e) => eprintln!("skip {:?}: {}", path, e),
            }
        }
        batch
    }
}

impl Iterator for DataLoader {
    type Item = Vec<Sample>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr_index <= self.num_batches() {
            let batch = self.get_batch(self.curr_index);
            self.curr_index += 1;
            Some(batch)
        } else {
            None
        }
    }
}
