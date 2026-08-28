use super::layer;

pub struct Network {
    layers: Vec<Box<dyn layer::Layer>>,
}

impl Network {
    pub fn new(layers: Vec<Box<dyn layer::Layer>>) -> Self {
        Self { layers }
    }

    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut out: Vec<Vec<f32>> = x.to_vec();
        for layer in self.layers.iter_mut() {
            out = layer.forward(&out);
        }
        out
    }

    pub fn backward(&mut self, grad: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut g: Vec<Vec<f32>> = grad.to_vec();
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
