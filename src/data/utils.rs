use super::structures::{FloatImageVector, IntImageVector, Sample};
use image::ImageReader;
use std::fs;
use std::path::{Path, PathBuf};

pub fn read_image(path: &Path) -> Result<IntImageVector, Box<dyn std::error::Error>> {
    let img = ImageReader::open(path)?.decode()?.to_luma8();
    Ok(img.into_raw())
}

pub fn normalize(pixels: IntImageVector) -> FloatImageVector {
    pixels.iter().map(|p: &u8| *p as f32 / 255.0).collect()
}

pub fn read_label(path: &Path) -> Result<u8, Box<dyn std::error::Error>> {
    let label: u8 = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .ok_or("cannot extract label from path")?
        .parse()?; // "0" -> 0u8
    Ok(label)
}

pub fn load_sample(path: impl AsRef<Path>) -> Result<Sample, Box<dyn std::error::Error>> {
    let path = path.as_ref();
    let image = read_image(path)?;
    let image = normalize(image);
    let label = read_label(path)?;
    Ok(Sample { image, label })
}

pub fn collect_paths(root: impl AsRef<Path>) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut paths = Vec::new();

    for label_entry in fs::read_dir(root)? {
        let label_dir = label_entry?.path();
        if !label_dir.is_dir() {
            continue;
        }
        for img_entry in fs::read_dir(&label_dir)? {
            let img_path = img_entry?.path();
            if img_path.extension().and_then(|e| e.to_str()) == Some("png") {
                paths.push(img_path);
            }
        }
    }

    Ok(paths)
}
