use super::utils;
use ndarray::{Array2, Axis};

pub trait Loss {
    fn loss(&self, logits: &Array2<f32>, true_values: &[usize]) -> f32;
    fn grad(&self, logits: &Array2<f32>, true_values: &[usize]) -> Array2<f32>;
}

#[derive(Default)]
pub struct SoftmaxCrossEntropyLoss;

impl SoftmaxCrossEntropyLoss {
    fn get_probas_from_logits(&self, logits: &Array2<f32>) -> Array2<f32> {
        utils::softmax(logits)
    }
}

impl Loss for SoftmaxCrossEntropyLoss {
    fn loss(&self, logits: &Array2<f32>, true_values: &[usize]) -> f32 {
        let probas: Array2<f32> = self.get_probas_from_logits(logits);
        let batch = probas.nrows();
        let mut loss: f32 = 0.0;

        for b in 0..batch {
            let p = probas[[b, true_values[b]]].max(1e-7); // ndarray-индексация [[row, col]]
            loss += -p.ln();
        }
        loss / batch as f32
    }
    fn grad(&self, logits: &Array2<f32>, true_values: &[usize]) -> Array2<f32> {
        let probas: Array2<f32> = self.get_probas_from_logits(logits);
        let batch: usize = probas.nrows();

        let mut grad: Array2<f32> = probas;

        for b in 0..batch {
            grad[[b, true_values[b]]] -= 1.0;
        }
        grad / batch as f32
    }
}
