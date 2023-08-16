use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std:: error;
use ndarray::Array2;
use ndarray_npy::WriteNpyExt;
use std::io::BufWriter;


pub type FormatTraces = i16;
pub type FormatMetadata = i32;

pub fn read_leakages(dir: &str)-> Result<Array2<FormatTraces>, Box<dyn error::Error>>{
    let reader: File = File::open(dir)?;
    let arr: ndarray::ArrayBase<ndarray::OwnedRepr<FormatTraces>, ndarray::Dim<[usize; 2]>> = Array2::<FormatTraces>::read_npy(reader)?;
    Ok(arr)
}           

pub fn read_metadata(dir: &str)-> Result<Array2<FormatMetadata>, Box<dyn error::Error>>{
    let reader: File = File::open(dir)?;
    let arr: ndarray::ArrayBase<ndarray::OwnedRepr<FormatMetadata>, ndarray::Dim<[usize; 2]>> = Array2::<FormatMetadata>::read_npy(reader)?;
    Ok(arr)
}


pub fn write_npy(dir: &str, ar: Array2<f32>)-> Result<(), Box<dyn error::Error>>{
    let writer = BufWriter::new(File::create(dir)?);
    let _ = ar.write_npy(writer);
    Ok(())
}



