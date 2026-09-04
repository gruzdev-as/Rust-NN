use ndarray::Array2;

pub trait Layer {
    fn forward(&mut self, x: &Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, grad_out: &Array2<f32>) -> Array2<f32>;
    fn update(&mut self, lr: f32);
}
