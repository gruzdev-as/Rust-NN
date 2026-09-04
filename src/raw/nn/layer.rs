pub trait Layer {
    fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>>;
    fn backward(&mut self, grad_out: &[Vec<f32>]) -> Vec<Vec<f32>>;
    fn update(&mut self, lr: f32);
}
