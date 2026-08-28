use std::path::PathBuf;

mod data; // __init__ 
use data::{DataLoader, Dataset, Sample}; // import

mod nn;
use nn::{Layer, LinearLayer, Loss, Network, ReLU, SoftmaxCrossEntropyLoss};

mod metrics;

const TRAIN_ROOT: &str = "data/mnist_png/train";
const TEST_ROOT: &str = "data/mnist_png/test";
const BS: usize = 64;
const LR: f32 = 0.001;
const NUM_EPOCHS: i32 = 10;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut net: Network = build_nn();
    let loss: SoftmaxCrossEntropyLoss = SoftmaxCrossEntropyLoss::default();
    let mut train_dataloader: DataLoader = init_dataloader(TRAIN_ROOT, BS, true)?;
    let mut test_dataloader: DataLoader = init_dataloader(TEST_ROOT, BS, false)?;

    for epoch in 0..NUM_EPOCHS {
        println!("=== EPOCH {} of {NUM_EPOCHS} ===", epoch + 1);
        train_one_epoch(&mut net, &loss, &mut train_dataloader);
        validate_one_epoch(&mut net, &mut test_dataloader);
    }
    Ok(())
}

fn build_nn() -> Network {
    Network::new(vec![
        Box::new(LinearLayer::new(784, 128)),
        Box::new(ReLU::new()),
        Box::new(LinearLayer::new(128, 10)),
    ])
}

fn train_one_epoch(net: &mut Network, loss: &impl Loss, loader: &mut DataLoader) {
    let total_train_batches = loader.num_batches();
    let mut running_loss: f32 = 0.0;
    for (idx, batch) in loader.by_ref().enumerate() {
        let labels: Vec<u8> = batch.iter().map(|sample| sample.label.clone()).collect();
        let input: Vec<Vec<f32>> = batch.iter().map(|sample| sample.image.clone()).collect();

        let logits: Vec<Vec<f32>> = net.forward(&input);
        running_loss += loss.loss(&logits, &labels);
        let grad: Vec<Vec<f32>> = loss.grad(&logits, &labels);
        net.backward(&grad);
        net.update(LR);

        if idx % 50 == 0 {
            println!("batch {idx}/{total_train_batches}  loss {:.4}", running_loss / 50.0);
            running_loss = 0.0
        }
    }
    loader.reset();
}

fn validate_one_epoch(net: &mut Network, loader: &mut DataLoader) {
    let mut predictions: Vec<usize> = Vec::new();
    let mut true_values: Vec<usize> = Vec::new();

    for (idx, batch) in loader.by_ref().enumerate() {
        let labels: Vec<usize> = batch
            .iter()
            .map(|sample| sample.label.clone())
            .map(|x| x as usize)
            .collect();
        let input: Vec<Vec<f32>> = batch.iter().map(|sample| sample.image.clone()).collect();

        let predicted: Vec<usize> = net
            .forward(&input)
            .iter()
            .map(|row| nn::utils::argmax(row).unwrap())
            .collect();

        true_values.extend_from_slice(&labels);
        predictions.extend_from_slice(&predicted);
    }
    loader.reset();
    let accuracy = metrics::classification::accuracy_score(&true_values, &predictions);
    println!("Accuracy: {}", accuracy)
}

fn init_dataloader(root: &str, bs: usize, shuffle: bool) -> Result<DataLoader, Box<dyn std::error::Error>> {
    let paths: Vec<PathBuf> = data::utils::collect_paths(root)?;
    let dataset: Dataset = Dataset::new(paths);
    Ok(DataLoader::new(dataset, bs, shuffle))
}
