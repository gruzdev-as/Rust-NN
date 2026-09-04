use super::layer::Layer;
use ndarray::Array2;

pub struct ReLU {
    last_input: Array2<f32>,
}

impl ReLU {
    pub fn new() -> Self {
        Self {
            last_input: Array2::zeros((0, 0)),
        }
    }
}

impl Layer for ReLU {
    fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        self.last_input = x.clone();
        let mut output: Array2<f32> = x.clone();
        output.mapv_inplace(|val| val.max(0.0));
        output
    }
    fn backward(&mut self, grad_out: &Array2<f32>) -> Array2<f32> {
        let mut grad = grad_out.clone();
        grad.zip_mut_with(&self.last_input, |g, &inp| {
            if inp <= 0.0 {
                *g = 0.0;
            }
        });
        grad
    }
    fn update(&mut self, _lr: f32) {
        // параметров нет, верх запян
    }
}
