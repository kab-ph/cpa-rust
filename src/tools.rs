use ndarray_npy::ReadNpyExt;
use std::fs::File;
use ndarray::{Array2, ArrayView2};
use ndarray_npy::WriteNpyExt;
use std::io::BufWriter;
use ndarray_npy::ReadableElement;


pub fn read_array_2_from_npy_file<T: ReadableElement> (dir: &str)-> Array2<T>{
    let reader: File = File::open(dir).unwrap();
    let arr: Array2<T> = Array2::<T>::read_npy(reader).unwrap();
    arr
}           


pub fn write_npy(dir: &str, ar: ArrayView2<f32>){
    let writer = BufWriter::new(File::create(dir).unwrap());
    let _ = ar.write_npy(writer);
}
