use std::path::PathBuf;

mod dataloader; // __init__ 
use dataloader::{DataLoader, Dataset, Sample}; // import

mod layer;
use layer::LinearLayer;

const BS: usize = 512;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let train_paths: Vec<PathBuf> = dataloader::collect_paths("data/mnist_png/train")?;
    let test_paths: Vec<PathBuf> = dataloader::collect_paths("data/mnist_png/test")?;
    println!(
        "found {} train samples and {} test samples",
        train_paths.len(),
        test_paths.len()
    );
    let train_dataset: Dataset = Dataset::new(train_paths);
    let test_dataset: Dataset = Dataset::new(test_paths);

    let mut train_dataloader: DataLoader = DataLoader::new(train_dataset, BS, true);
    let mut test_dataloader: DataLoader = DataLoader::new(test_dataset, BS, false);

    let total_train_batches: usize = train_dataloader.num_batches();
    let total_test_batches: usize = test_dataloader.num_batches();

    for (idx, batch) in (&mut train_dataloader).enumerate() {
        println!("TRAIN: Batch {idx} of {total_train_batches}, {} samples", batch.len());
    }

    train_dataloader.reset();

    for (idx, batch) in (&mut test_dataloader).enumerate() {
        println!("TEST: Batch {idx} of {total_test_batches}, {} samples", batch.len());
    }

    test_dataloader.reset();

    let batch: Vec<Sample> = train_dataloader.next().unwrap();
    let sample: &Sample = &batch[0];
    let layer: LinearLayer = LinearLayer::new(24 * 24, 10);
    let output: Vec<f32> = layer.forward(&sample.image);

    println!("выход слоя: {:?}", output);
    println!("истинная метка: {}", sample.label);

    Ok(())
}
