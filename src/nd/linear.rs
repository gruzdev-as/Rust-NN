use super::layer::Layer;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub struct LinearLayer {
    num_features_in: usize,
    num_features_out: usize,

    w: Array2<f32>,
    b: Array1<f32>,

    grad_w: Array2<f32>,
    grad_b: Array1<f32>,
    last_input: Array2<f32>,
}

impl LinearLayer {
    pub fn new(num_features_in: usize, num_features_out: usize) -> Self {
        let a: f32 = (6.0 / num_features_in as f32).sqrt(); // Имитируем оч плохо Kaiming uniform TODO: fix.

        let w: Array2<f32> = Array2::random((num_features_in, num_features_out), Uniform::new(-a, a).unwrap());
        let grad_w: Array2<f32> = Array2::zeros((num_features_in, num_features_out));

        let b: Array1<f32> = Array1::zeros(num_features_out);
        let grad_b: Array1<f32> = Array1::zeros(num_features_out);

        let last_input: Array2<f32> = Array2::zeros((0, 0));

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
    fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        self.last_input = x.clone();
        &x.dot(&self.w) + &self.b
    }
    fn backward(&mut self, grad_out: &Array2<f32>) -> Array2<f32> {
        self.grad_w = self.last_input.t().dot(grad_out);
        self.grad_b = grad_out.sum_axis(Axis(0));

        grad_out.dot(&self.w.t())
    }
    fn update(&mut self, lr: f32) {
        self.w = &self.w - lr * &self.grad_w;
        self.b = &self.b - lr * &self.grad_b;
    }
}
