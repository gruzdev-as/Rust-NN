use image::ImageReader;
use rand::seq::SliceRandom;
use std::cmp::min;
use std::fs;
use std::path::{Path, PathBuf};

pub type IntImageVector = Vec<u8>;
pub type FloatImageVector = Vec<f32>;

pub struct Sample {
    pub image: FloatImageVector,
    pub label: u8,
}

pub struct Dataset {
    pub paths: Vec<PathBuf>,
}

pub struct DataLoader {
    dataset: Dataset,
    bs: usize,
    shuffle: bool,
    indices: Vec<usize>,
    curr_index: usize,
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
            match load_sample(path) {
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

fn read_image(path: &Path) -> Result<IntImageVector, Box<dyn std::error::Error>> {
    let img = ImageReader::open(path)?.decode()?.to_luma8();
    Ok(img.into_raw())
}

fn normalize(pixels: IntImageVector) -> FloatImageVector {
    pixels.iter().map(|p: &u8| *p as f32 / 255.0).collect()
}

fn read_label(path: &Path) -> Result<u8, Box<dyn std::error::Error>> {
    let label: u8 = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .ok_or("cannot extract label from path")?
        .parse()?; // "0" -> 0u8
    Ok(label)
}

pub fn load_sample(path: impl AsRef<Path>) -> Result<Sample, Box<dyn std::error::Error>> {
    let path = path.as_ref();
    let image = read_image(path)?;
    let image = normalize(image);
    let label = read_label(path)?;
    Ok(Sample { image, label })
}

pub fn collect_paths(root: impl AsRef<Path>) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut paths = Vec::new();

    for label_entry in fs::read_dir(root)? {
        let label_dir = label_entry?.path();
        if !label_dir.is_dir() {
            continue;
        }
        for img_entry in fs::read_dir(&label_dir)? {
            let img_path = img_entry?.path();
            if img_path.extension().and_then(|e| e.to_str()) == Some("png") {
                paths.push(img_path);
            }
        }
    }

    Ok(paths)
}
