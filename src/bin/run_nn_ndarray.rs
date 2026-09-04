use clap::Parser;
use ndarray::{Array, Array2, Axis};
use std::path::PathBuf;

use rust_nn::data::utils as data_utils;
use rust_nn::data::{Config, DataLoader, Dataset};
use rust_nn::nd::metrics;
use rust_nn::nd::nn::utils as nn_utils;
use rust_nn::nd::nn::{LinearLayer, Loss, Network, ReLU, SoftmaxCrossEntropyLoss};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config: Config = Config::parse();

    let mut net: Network = build_nn();
    let loss: SoftmaxCrossEntropyLoss = SoftmaxCrossEntropyLoss::default();
    let mut train_dataloader: DataLoader = init_dataloader(&config.train_data_folder, config.batch_size, true)?;
    let mut test_dataloader: DataLoader = init_dataloader(&config.test_data_folder, config.batch_size, false)?;

    for epoch in 0..config.num_epochs {
        println!("=== EPOCH {} of {} ===", epoch + 1, config.num_epochs);
        train_one_epoch(&mut net, &loss, &mut train_dataloader, config.lr);
        validate_one_epoch(&mut net, &mut test_dataloader);
    }
    Ok(())
}

fn build_nn() -> Network {
    Network::new(vec![
        Box::new(LinearLayer::new(784, 128)),
        Box::new(ReLU::new()),
        Box::new(LinearLayer::new(128, 128)),
        Box::new(ReLU::new()),
        Box::new(LinearLayer::new(128, 10)),
    ])
}

fn train_one_epoch(net: &mut Network, loss: &impl Loss, loader: &mut DataLoader, lr: f32) {
    let total_train_batches = loader.num_batches();
    let mut running_loss: f32 = 0.0;
    for (idx, batch) in loader.by_ref().enumerate() {
        let labels: Vec<usize> = batch.iter().map(|sample| sample.label.clone() as usize).collect();
        let input: Vec<Vec<f32>> = batch.iter().map(|sample| sample.image.clone()).collect();

        let rows: usize = input.len();
        let cols: usize = input.first().map_or(0, |x| x.len());
        let flat: Vec<f32> = input.into_iter().flatten().collect();

        let input: Array2<f32> = Array::from_shape_vec((rows, cols), flat).unwrap();
        let logits: Array2<f32> = net.forward(&input);

        running_loss += loss.loss(&logits, &labels);

        let grad: Array2<f32> = loss.grad(&logits, &labels);
        net.backward(&grad);
        net.update(lr);

        if idx % 200 == 0 && idx != 0 {
            println!("batch {idx}/{total_train_batches}  loss {:.4}", running_loss / 200.0);
            running_loss = 0.0
        }
    }
    loader.reset();
}

fn validate_one_epoch(net: &mut Network, loader: &mut DataLoader) {
    let mut predictions: Vec<usize> = Vec::new();
    let mut true_values: Vec<usize> = Vec::new();

    for batch in loader.by_ref() {
        let labels: Vec<usize> = batch
            .iter()
            .map(|sample| sample.label.clone())
            .map(|x| x as usize)
            .collect();
        let input: Vec<Vec<f32>> = batch.iter().map(|sample| sample.image.clone()).collect();

        let rows: usize = input.len();
        let cols: usize = input.first().map_or(0, |x| x.len());
        let flat: Vec<f32> = input.into_iter().flatten().collect();

        let input: Array2<f32> = Array::from_shape_vec((rows, cols), flat).unwrap();
        let predicted: Vec<usize> = net
            .forward(&input)
            .axis_iter(Axis(0))
            .map(|row| nn_utils::argmax(row.as_slice().unwrap()).unwrap())
            .collect();

        true_values.extend_from_slice(&labels);
        predictions.extend_from_slice(&predicted);
    }
    loader.reset();
    let accuracy = metrics::classification::accuracy_score(&true_values, &predictions);
    println!("Accuracy: {}", accuracy)
}

fn init_dataloader(root: &str, bs: usize, shuffle: bool) -> Result<DataLoader, Box<dyn std::error::Error>> {
    let paths: Vec<PathBuf> = data_utils::collect_paths(root)?;
    let dataset: Dataset = Dataset::new(paths);
    Ok(DataLoader::new(dataset, bs, shuffle))
}
