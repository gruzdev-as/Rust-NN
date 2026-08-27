use std::path::PathBuf;

mod data; // __init__ 
use data::{DataLoader, Dataset, Sample}; // import

mod nn;
use nn::{Layer, LinearLayer, ReLU};

const TRAIN_ROOT: &str = "data/mnist_png/train";
const TEST_ROOT: &str = "data/mnist_png/test";
const BS: usize = 512;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut train_dataloader = init_dataloader(TRAIN_ROOT, BS, true)?;
    let mut test_dataloader = init_dataloader(TEST_ROOT, BS, false)?;

    run_one_epoch(&mut train_dataloader);
    run_one_epoch(&mut test_dataloader);

    let batch: Vec<Sample> = train_dataloader.next().unwrap();
    let labels: Vec<u8> = batch.iter().map(|sample| sample.label.clone()).collect();
    let images: Vec<Vec<f32>> = batch.iter().map(|sample| sample.image.clone()).collect();
    
    let mut layer: LinearLayer = LinearLayer::new(24 * 24, 10);
    let mut relu: ReLU = ReLU::new();
    
    let mut output: Vec<Vec<f32>> = layer.forward(&images);
    output = relu.forward(&output);

    println!("выход слоя: {:?}", output);
    println!("истинные метки: {:?}", labels);

    Ok(())
}

fn run_one_epoch(loader: &mut DataLoader) {
    let total_train_batches: usize = loader.num_batches();
    for (idx, batch) in loader.enumerate() {
        println!("Batch {idx} of {total_train_batches}, {} samples", batch.len());
    }
    loader.reset();
}

fn init_dataloader(root: &str, bs: usize, shuffle: bool) -> Result<DataLoader, Box<dyn std::error::Error>> {
    let paths: Vec<PathBuf> = data::utils::collect_paths(root)?;
    let dataset: Dataset = Dataset::new(paths);
    Ok(DataLoader::new(dataset, bs, shuffle))
}
