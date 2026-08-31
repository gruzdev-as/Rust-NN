use clap::Parser;

pub type IntImageVector = Vec<u8>;
pub type FloatImageVector = Vec<f32>;

pub struct Sample {
    pub image: FloatImageVector,
    pub label: u8,
}

#[derive(Parser)]
pub struct Config {
    #[arg(long, default_value = "data/mnist_png/train")]
    pub train_data_folder: String,

    #[arg(long, default_value = "data/mnist_png/test")]
    pub test_data_folder: String,

    #[arg(long, default_value_t = 64)]
    pub batch_size: usize,

    #[arg(long, default_value_t = 5)]
    pub num_epochs: usize,

    #[arg(long, default_value_t = 0.001)]
    pub lr: f32,
}
