use rand::RngExt;

pub struct LinearLayer {
    num_features_in: usize,
    num_features_out: usize,
    w: Vec<Vec<f32>>, // w[i][j]: вес входа j для выхода i
    b: Vec<f32>,      // b[i]: смещение выхода i

                      // grad: Vec<Vec<f32>> ,
                      // anc: &LinearLayer,
}

impl LinearLayer {
    pub fn new(num_features_in: usize, num_features_out: usize) -> Self {
        let mut rng = rand::rng();

        let mut w: Vec<Vec<f32>> = Vec::new();
        for _ in 0..num_features_out {
            let mut row: Vec<f32> = Vec::new();
            for _ in 0..num_features_in {
                row.push(rng.random_range(-1.0..1.0));
            }
            w.push(row);
        }

        let b: Vec<f32> = vec![0.0; num_features_out];

        Self {
            num_features_in,
            num_features_out,
            w,
            b,
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut out: Vec<f32> = Vec::new();

        for i in 0..self.num_features_out {
            let mut sum: f32 = self.b[i];
            for j in 0..self.num_features_in {
                sum += self.w[i][j] * x[j];
            }
            out.push(sum);
        }
        out
    }
}
