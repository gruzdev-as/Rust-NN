use super::layer::Layer;
use super::utils;
use rand::RngExt;

pub struct LinearLayer {
    num_features_in: usize,
    num_features_out: usize,
    w: Vec<Vec<f32>>, // w[i][j]: вес входа j для выхода i
    b: Vec<f32>,      // b[i]: смещение выхода i

    grad_w: Vec<Vec<f32>>, // ∂loss/∂w[i][j]
    grad_b: Vec<f32>,      // ∂loss/∂b[i]
    last_input: Vec<Vec<f32>>,
}

impl LinearLayer {
    pub fn new(num_features_in: usize, num_features_out: usize) -> Self {
        let mut rng = rand::rng();

        let w = utils::make_matrix(num_features_out, num_features_in, || rng.random_range(-1.0..1.0));
        let grad_w = utils::make_matrix(num_features_out, num_features_in, || 0.0);

        let b: Vec<f32> = vec![0.0; num_features_out];
        let grad_b: Vec<f32> = vec![0.0; num_features_out];

        let last_input: Vec<Vec<f32>> = Vec::new();

        Self {
            num_features_in,
            num_features_out,
            w,
            b,
            grad_w,
            grad_b,
            last_input,
        }
    }
}

impl Layer for LinearLayer {
    fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.last_input = x.to_vec();

        let mut out: Vec<Vec<f32>> = Vec::new();
        for sample in x {
            let mut out_row = Vec::new();
            for i in 0..self.num_features_out {
                let mut sum = self.b[i];
                for j in 0..self.num_features_in {
                    sum += self.w[i][j] * sample[j];
                }
                out_row.push(sum);
            }
            out.push(out_row);
        }
        out
    }

    fn backward(&mut self, grad_out: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let batch = grad_out.len();

        self.grad_w = utils::make_matrix(self.num_features_out, self.num_features_in, || 0.0);
        self.grad_b = vec![0.0; self.num_features_out];

        for b in 0..batch {
            for i in 0..self.num_features_out {
                self.grad_b[i] += grad_out[b][i];
            }
        }

        for b in 0..batch {
            for i in 0..self.num_features_out {
                for j in 0..self.num_features_in {
                    self.grad_w[i][j] += grad_out[b][i] * self.last_input[b][j];
                }
            }
        }

        let mut grad_input = utils::make_matrix(batch, self.num_features_in, || 0.0);
        for b in 0..batch {
            for j in 0..self.num_features_in {
                for i in 0..self.num_features_out {
                    grad_input[b][j] += grad_out[b][i] * self.w[i][j];
                }
            }
        }
        grad_input
    }
}
