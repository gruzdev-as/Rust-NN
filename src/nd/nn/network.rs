use super::layer;
use ndarray::Array2;

pub struct Network {
    layers: Vec<Box<dyn layer::Layer>>,
}
impl Network {
    pub fn new(layers: Vec<Box<dyn layer::Layer>>) -> Self {
        Self { layers }
    }

    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        let mut out: Array2<f32> = x.clone();
        for layer in self.layers.iter_mut() {
            out = layer.forward(&out);
        }
        out
    }

    pub fn backward(&mut self, grad: &Array2<f32>) -> Array2<f32> {
        let mut g: Array2<f32> = grad.clone();
        for layer in self.layers.iter_mut().rev() {
            g = layer.backward(&g);
        }
        g
    }

    pub fn update(&mut self, lr: f32) {
        for layer in self.layers.iter_mut() {
            layer.update(lr);
        }
    }
}
