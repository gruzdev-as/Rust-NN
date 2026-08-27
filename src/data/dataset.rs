use std::path::{Path, PathBuf};

pub struct Dataset {
    pub paths: Vec<PathBuf>,
}

impl Dataset {
    pub fn new(paths: Vec<PathBuf>) -> Self {
        Self { paths }
    }
    pub fn len(&self) -> usize {
        self.paths.len()
    }
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }
    pub fn get(&self, idx: usize) -> &Path {
        &self.paths[idx]
    }
}
