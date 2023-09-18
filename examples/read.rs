use ndarray_npy::ReadNpyExt;
use std::fs::File;
use ndarray::{Array2, ArrayView2};
use ndarray_npy::WriteNpyExt;
use std::io::BufWriter;


pub type FormatTraces = i16;
pub type FormatMetadata = i32;


pub fn read_leakages(dir: &str)-> Array2<FormatTraces>{
    let reader: File = File::open(dir).unwrap();
    let arr: Array2<FormatTraces> = Array2::<FormatTraces>::read_npy(reader).unwrap();
    arr
}           

pub fn read_metadata(dir: &str)-> Array2<FormatMetadata>{
    let reader: File = File::open(dir).unwrap();
    let arr:Array2<FormatMetadata> = Array2::<FormatMetadata>::read_npy(reader).unwrap();
    arr
}


pub fn write_npy(dir: &str, ar: ArrayView2<f32>){
    let writer = BufWriter::new(File::create(dir).unwrap());
    let _ = ar.write_npy(writer);
}

#[allow(dead_code)]
fn main(){}