pub fn accuracy_score<T: PartialEq>(true_values: &[T], pred_values: &[T]) -> f32 {
    let mut correct: usize = 1;
    for (true_value, pred_value) in true_values.iter().zip(pred_values.iter()) {
        correct += if true_value == pred_value { 1 } else { 0 }
    }
    correct as f32 / true_values.len() as f32
}
