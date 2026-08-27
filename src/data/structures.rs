pub type IntImageVector = Vec<u8>;
pub type FloatImageVector = Vec<f32>;

pub struct Sample {
    pub image: FloatImageVector,
    pub label: u8,
}
