use simple_bar::ProgressBar;
use rayon::prelude::{ParallelIterator, ParallelBridge};
use std::time::Instant;
use ndarray::*;
use cpa::cpa::*;
use cpa::leakage::{hw, sbox};
use cpa::tools::{read_array_2_from_npy_file, write_npy};


// traces format
type FormatTraces = i16;
type FormatMetadata = i32;

// leakage model
pub fn leakage_model(value: usize, guess: usize) -> usize{
    hw(sbox[(value ^ guess) as usize] as usize)
}

// multi-threading cpa
fn cpa()
{
    let size: usize = 5000;  // Number of samples 
    let guess_range = 256;  // 2**(key length)
    let target_byte = 1;    
    let folder = String::from("data"); // Directory of leakages and metadata
    let nfiles = 5; // Number of files in the directory. TBD: Automating this value
    let mut bar = ProgressBar::default(nfiles as u32, 50, false);
    
    /* Parallel operation using multi-threading on patches */
    let mut cpa: Cpa = (0..nfiles).into_iter().
    map(|n| {
        bar.update();
        let dir_l: String = format!("{}{}{}{}", folder, "/l", n.to_string(), ".npy" );
        let dir_p = format!("{}{}{}{}", folder, "/p", n.to_string(), ".npy");
        let leakages: Array2<FormatTraces>= read_array_2_from_npy_file::<FormatTraces>(&dir_l);
        let plaintext: Array2<FormatMetadata> = read_array_2_from_npy_file::<FormatMetadata>(&dir_p);
        (leakages, plaintext)
        
    }).into_iter().par_bridge().map(|patch|
    {   
        let mut c: Cpa = Cpa::new(size, guess_range, target_byte, leakage_model);
        let len_leakage = patch.0.shape()[0];       
        for i in 0..len_leakage{
            c.update(
                patch.0.row(i).map(|x| *x as usize), 
                patch.1.row(i).map(|y| *y as usize)
            );      
        }    
        c
    }).reduce(|| Cpa::new(size, guess_range, target_byte, leakage_model), |a: Cpa, b| a+b);
    cpa.finalize();
    println!("Guessed key = {}", cpa.pass_guess());
    // save corr key curves in npy
    // write_npy("examples/corr.npy", cpa.pass_corr_array().view());
    
}




fn rank(){
    let size: usize = 5000; // Number of samples 
    let guess_range = 256; // 2**(key length)
    let target_byte = 1;
    let folder = String::from("data");  
    let nfiles = 5;   
    let mut bar = ProgressBar::default(nfiles as u32, 50, false);
    let chunk = 3000;
    let mut rank: Cpa = Cpa::new(size, guess_range, target_byte, leakage_model);
    for file in 0..nfiles{
        let dir_l = format!("{}{}{}{}", folder, "/l", file.to_string(), ".npy" ); // leakage directory
        let dir_p = format!("{}{}{}{}", folder, "/p", file.to_string(), ".npy"); // plaintext directory
        let leakages: Array2<FormatTraces> = read_array_2_from_npy_file::<FormatTraces>(&dir_l);
        let plaintext: Array2<FormatMetadata> = read_array_2_from_npy_file::<FormatMetadata>(&dir_p);
        let len_file = leakages.shape()[0];
        for sample in (0..len_file).step_by(chunk){
            let l_sample: ndarray::ArrayBase<ndarray::ViewRepr<&FormatTraces>, ndarray::Dim<[usize; 2]>> = leakages.slice(s![sample..sample+chunk, ..]);
            let p_sample = plaintext.slice(s![sample..sample+chunk, ..]);
            let x = (0..chunk).into_iter().par_bridge().
            fold(|| Cpa::new(size, guess_range, target_byte, leakage_model), |mut r: Cpa, n|{
                r.update(l_sample.row(n).map(|l: &FormatTraces| *l as usize),
                         p_sample.row(n).map(|p: &FormatMetadata| *p as usize));
                r
            }).reduce(||Cpa::new(size, guess_range, target_byte, leakage_model), |lhs, rhs| lhs + rhs);
            rank = rank + x;
            rank.finalize();
        
        }
        bar.update(); 
    }
    // save rank key curves in npy
    write_npy("examples/rank.npy", rank.pass_rank());
}
    
    
fn main(){
    let  mut t = Instant::now();
    cpa();
    println!("Time for CPA {:?}", t.elapsed());
    t = Instant::now();
    rank();
    println!("Time for key rank {:?}", t.elapsed());
    println!("l_b");
}
