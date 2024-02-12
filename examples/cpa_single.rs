use cpa::cpa_single::*;
use cpa::leakage::{hw, sbox};
use cpa::tools::{read_array_2_from_npy_file, write_array};
use indicatif::ProgressIterator;
use ndarray::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::time::{self};

// traces format
type FormatTraces = i16;
type FormatMetadata = i32;

// leakage model
pub fn leakage_model(value: ArrayView1<usize>, guess: usize) -> usize {
    hw(sbox((value[1] ^ guess) as u8) as usize)
}

fn cpa() {
    let start_sample: usize = 0;
    let end_sample: usize = 1000;
    let size: usize = end_sample - start_sample; // Number of samples
    let guess_range: i32 = 256; // 2**(key length)
    let folder = String::from("../data/old"); // Directory of leakages and metadata
    let nfiles: i32 = 5; // Number of files in the directory. TBD: Automating this value
    let mut cpa = Cpa::new(size, guess_range, leakage_model);

    for i in (0..nfiles).progress() {
        let dir_l = format!("{folder}/l{i}.npy");
        let dir_p = format!("{folder}/p{i}.npy");
        let leakages: Array2<FormatTraces> = read_array_2_from_npy_file::<FormatTraces>(&dir_l);
        let plaintext: Array2<FormatMetadata> =
            read_array_2_from_npy_file::<FormatMetadata>(&dir_p);
        let len_leakages = leakages.shape()[0];
        for row in 0..len_leakages{
            let sample_trace: Array1<usize> = leakages.row(row).slice(s![start_sample..end_sample]).map(|l| *l as usize);
            let sample_metadat: Array1<usize> = plaintext.row(row).map(|p| *p as usize );
            cpa.update(sample_trace, sample_metadat);
        }
    }

    cpa.finalize();
    println!("Guessed key = {}", cpa.pass_guess());
    // save corr key curves in npy
    write_array("../results/corr.npy", cpa.pass_corr_array().view());
}



fn cpa_parallel() {
    let start_sample: usize = 0;
    let end_sample: usize = 1000;
    let size: usize = end_sample - start_sample; // Number of samples
    let guess_range = 256; // 2**(key length)
    let folder = String::from("../data/old"); // Directory of leakages and metadata
    let nfiles: i32 = 5; // Number of files in the directory. TBD: Automating this value
    let mut cpa_parallel = (0..nfiles).into_par_iter().map(|num|{
        let dir_l = format!("{folder}/l{num}.npy");
        let dir_p = format!("{folder}/p{num}.npy");
        let leakages: Array2<FormatTraces> = read_array_2_from_npy_file::<FormatTraces>(&dir_l);
        let plaintext: Array2<FormatMetadata> =
            read_array_2_from_npy_file::<FormatMetadata>(&dir_p);
        let len_leakages = leakages.shape()[0];
        let mut cpa = Cpa::new(size, guess_range, leakage_model);
        for row in 0..len_leakages{
            let sample_trace: Array1<usize> = leakages.row(row).slice(s![start_sample..end_sample]).map(|l| *l as usize);
            let sample_metadat: Array1<usize> = plaintext.row(row).map(|p| *p as usize );
            cpa.update(sample_trace, sample_metadat);
        }
        cpa
    }).reduce(||Cpa::new(size, guess_range, leakage_model), |x, y| x + y);

    cpa_parallel.finalize();
    println!("Guessed key = {}", cpa_parallel.pass_guess());
    // save corr key curves in npy
    write_array("../results/corr.npy", cpa_parallel.pass_corr_array().view());
}


fn main() {
    let mut t = time::Instant::now();
    cpa();
    println!("{:?}", t.elapsed());    
}


