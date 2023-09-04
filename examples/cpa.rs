use std:: error;
use simple_bar::ProgressBar;
use rayon::prelude::{ParallelIterator, ParallelBridge};
use std::time::Instant;
use ndarray::*;
use cpa::cpa::*;
use cpa::tools::{read_leakages, write_npy, read_metadata};
use cpa::leakage::{hw, sbox};


pub fn leakage_model(value: usize, guess: usize) -> usize{
    hw(sbox[(value ^ guess) as usize] as usize)
}


fn cpa_fast()
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
        let leakages = read_leakages(&dir_l);
        let plaintext = read_metadata(&dir_p);
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
    write_npy("corr.npy", cpa.finalize().view());

}


fn rank_modified(){
    let size: usize = 5000;
    let guess_range = 256;
    let target_byte = 1;
    let folder = String::from("data"); 
    let nfiles = 5;   
    let mut bar = ProgressBar::default(nfiles as u32, 50, false);
    let chunk = 3000;
    let mut rank: Cpa = Cpa::new(size, guess_range, target_byte, leakage_model);
    for file in 0..nfiles{
        
        let dir_l = format!("{}{}{}{}", folder, "/l", file.to_string(), ".npy" ); // leakage directory
        let dir_p = format!("{}{}{}{}", folder, "/p", file.to_string(), ".npy"); // plaintext directory
        let leakages = read_leakages(&dir_l);
        let plaintext = read_metadata(&dir_p);
        let len_file = leakages.shape()[0];
        for sample in (0..len_file).step_by(chunk){
            let l_sample: ndarray::ArrayBase<ndarray::ViewRepr<&i16>, ndarray::Dim<[usize; 2]>> = leakages.slice(s![sample..sample+chunk, ..]);
            let p_sample = plaintext.slice(s![sample..sample+chunk, ..]);
            let x = (0..chunk).into_iter().par_bridge().
            fold(|| Cpa::new(size, guess_range, target_byte, leakage_model), |mut r: Cpa, n|{
                r.update(l_sample.row(n).map(|l| *l as usize),
                         p_sample.row(n).map(|p| *p as usize));
                r
            }).reduce(||Cpa::new(size, guess_range, target_byte, leakage_model), |lhs, rhs| lhs + rhs);
            rank = rank + x;
            rank.finalize();
        
        }
        bar.update(); 
    }
    write_npy("rank_fast.npy", rank.pass_rank());
}
    
    
fn main()-> Result<(), Box<dyn error::Error>>{
    let  mut t = Instant::now();
    cpa_fast();
    println!("Slow  cpa time is {:?}", t.elapsed());
    t = Instant::now();
    rank_modified();
    println!("Fast cpa time is {:?}", t.elapsed());
    Ok(())
}




