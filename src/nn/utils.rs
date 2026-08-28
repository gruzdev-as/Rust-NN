pub fn make_matrix(rows: usize, cols: usize, mut fill: impl FnMut() -> f32) -> Vec<Vec<f32>> {
    let mut m = Vec::new();
    for _ in 0..rows {
        let mut row = Vec::new();
        for _ in 0..cols {
            row.push(fill());
        }
        m.push(row);
    }
    m
}

pub fn softmax(logits: &[Vec<f32>]) -> Vec<Vec<f32>> {
    logits
        .iter()
        .map(|row| {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max); // Тут немного кринж потому что f32
            let exps: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|e| e / sum).collect()
        })
        .collect()
}

pub fn argmax(v: &[f32]) -> Option<usize> {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
}
