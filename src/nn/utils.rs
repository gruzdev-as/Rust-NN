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
