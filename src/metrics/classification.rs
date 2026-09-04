use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

pub fn accuracy_score<T: PartialEq>(true_values: &[T], pred_values: &[T]) -> f32 {
    let mut correct: usize = 1;
    for (true_value, pred_value) in true_values.iter().zip(pred_values.iter()) {
        correct += if true_value == pred_value { 1 } else { 0 }
    }
    correct as f32 / true_values.len() as f32
}

fn get_metrics_binary(true_values: &[i8], pred_values: &[i8]) -> (f32, f32, f32) {
    let mut true_positive: f32 = 0.0;
    let mut false_positive: f32 = 0.0;
    let mut false_negative: f32 = 0.0;

    for (true_value, pred_value) in true_values.iter().zip(pred_values.iter()) {
        if *true_value == 1 && *pred_value == 1 {
            true_positive += 1.0;
        } else if *true_value == 1 && *pred_value == 0 {
            false_negative += 1.0;
        } else if *true_value == 0 && *pred_value == 1 {
            false_positive += 1.0;
        }
    }
    let precision = if true_positive + false_positive > 0.0 {
        true_positive / (true_positive + false_positive)
    } else {
        0.0
    };
    let recall = if true_positive + false_negative > 0.0 {
        true_positive / (true_positive + false_negative)
    } else {
        0.0
    };
    let f_score = if precision + recall > 0.0 {
        2.0 * recall * precision / (recall + precision)
    } else {
        0.0
    };

    (precision, recall, f_score)
}

pub fn get_per_class_metrics_macro<T: Eq + Hash + Clone + Ord>(true_values: &[T], pred_values: &[T]) -> (f32, f32, f32) {
    let unique_classes: HashSet<&T> = true_values.iter().collect();
    let mut per_class_map: HashMap<T, (f32, f32, f32)> = HashMap::new();

    for class in unique_classes.iter() {
        let true_binary: Vec<i8> = true_values.iter().map(|x| if x == *class { 1 } else { 0 }).collect();
        let pred_binary: Vec<i8> = pred_values.iter().map(|x| if x == *class { 1 } else { 0 }).collect();

        let metrics: (f32, f32, f32) = get_metrics_binary(&true_binary, &pred_binary);
        per_class_map.insert((*class).clone(), metrics);
    }

    let macro_precision: f32 =
        per_class_map.values().fold(0.0, |sum, &elem| sum + elem.0 as f32) / per_class_map.len() as f32;
    let macro_recall: f32 =
        per_class_map.values().fold(0.0, |sum, &elem| sum + elem.1 as f32) / per_class_map.len() as f32;
    let macro_f_score: f32 =
        per_class_map.values().fold(0.0, |sum, &elem| sum + elem.2 as f32) / per_class_map.len() as f32;

    (macro_precision, macro_recall, macro_f_score)
}
