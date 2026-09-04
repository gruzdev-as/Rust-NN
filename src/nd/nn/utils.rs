use ndarray::{Array1, Array2, Axis};

pub fn softmax(logits: &Array2<f32>) -> Array2<f32> {
    let max: Array1<f32> = logits.map_axis(Axis(1), |row| row.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    let max: Array2<f32> = max.insert_axis(Axis(1));

    let shifted: Array2<f32> = logits - &max;
    let exps: Array2<f32> = shifted.mapv(|v| v.exp());

    let sum: Array2<f32> = exps.sum_axis(Axis(1)).insert_axis(Axis(1));
    exps / &sum
}

pub fn argmax(v: &[f32]) -> Option<usize> {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
}
