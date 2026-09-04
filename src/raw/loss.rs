use super::utils;

pub trait Loss {
    fn loss(&self, logits: &[Vec<f32>], true_values: &[u8]) -> f32;
    fn grad(&self, logits: &[Vec<f32>], true_values: &[u8]) -> Vec<Vec<f32>>;
}

#[derive(Default)]
pub struct SoftmaxCrossEntropyLoss;

impl SoftmaxCrossEntropyLoss {
    fn get_probas_from_logits(&self, logits: &[Vec<f32>]) -> Vec<Vec<f32>> {
        utils::softmax(logits)
    }
}

impl Loss for SoftmaxCrossEntropyLoss {
    fn loss(&self, logits: &[Vec<f32>], true_values: &[u8]) -> f32 {
        let probas = self.get_probas_from_logits(logits);
        let batch = probas.len();
        let mut loss: f32 = 0.0;

        for i in 0..batch {
            let p = probas[i][true_values[i] as usize].max(1e-7); //  Чтоб не ушло в бесконечность
            loss += -p.ln();
        }
        loss / (batch as f32)
    }
    fn grad(&self, logits: &[Vec<f32>], true_values: &[u8]) -> Vec<Vec<f32>> {
        let probas = self.get_probas_from_logits(logits);
        let batch = probas.len();
        let features = probas[0].len();
        let mut grad: Vec<Vec<f32>> = Vec::new();

        for b in 0..batch {
            let mut row = Vec::new();
            for i in 0..features {
                let target = if i as u8 == true_values[b] { 1.0 } else { 0.0 };
                row.push((probas[b][i] - target) / batch as f32);
            }
            grad.push(row);
        }

        grad
    }
}
