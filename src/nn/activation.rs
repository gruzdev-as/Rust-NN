use super::layer::Layer;
use super::utils;

pub struct ReLU {
    last_input: Vec<Vec<f32>>,
}

impl ReLU {
    pub fn new() -> Self {
        Self { last_input: Vec::new() }
    }
}

impl Layer for ReLU {
    fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.last_input = x.to_vec();
        x.iter()
            .map(|row| row.iter().map(|&val| val.max(0.0)).collect())
            .collect()
    }
    fn backward(&mut self, grad_out: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let batch = grad_out.len();
        let features = grad_out[0].len();

        let mut grad_input = utils::make_matrix(batch, features, || 0.0);

        for b in 0..batch {
            for j in 0..features {
                if self.last_input[b][j] > 0.0 {
                    grad_input[b][j] = grad_out[b][j];
                }
            }
        }

        grad_input
    }
}
