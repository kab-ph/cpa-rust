use cpa::cpa_single::*;
use cpa::leakage::{hw, sbox};
use cpa::tools::{read_array_2_from_npy_file, write_array};
use indicatif::ProgressIterator;
use ndarray::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::time::{self};

// traces format
type FormatTraces = f64; // f64; //i16;
type FormatMetadata = u8; // u8; //i32;
static mut target_byte:usize = 0;
// leakage model

pub fn leakage_model(value: Array1<FormatMetadata>, guess: usize) -> usize {
    unsafe{
    hw(sbox(value[target_byte] as u8 ^ guess as u8) as usize)
}

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
         for row in 0..len_leakages {
             let sample_trace: Array1<f64> = leakages
                 .row(row)
                 .slice(s![start_sample..end_sample])
                 .map(|l| *l as f64);
             let sample_metadat: Array1<FormatMetadata> = plaintext.row(row).to_owned();
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
     let end_sample: usize = 5000;
     let size: usize = end_sample - start_sample; // Number of samples
     let guess_range = 256; // 2**(key length)
     let folder = String::from("../data/old"); // Directory of leakages and metadata
     let nfiles: i32 = 5; // Number of files in the directory. TBD: Automating this value
     let mut cpa_parallel = (0..nfiles)
         .into_par_iter()
         .map(|num| {
             let dir_l = format!("{folder}/l{num}.npy");
             let dir_p = format!("{folder}/p{num}.npy");
             let leakages: Array2<FormatTraces> = read_array_2_from_npy_file::<FormatTraces>(&dir_l);
             let plaintext: Array2<FormatMetadata> =
                 read_array_2_from_npy_file::<FormatMetadata>(&dir_p);
             let len_leakages = leakages.shape()[0];
             let mut cpa = Cpa::new(size, guess_range, leakage_model);
             for row in 0..len_leakages {
                 let sample_trace: Array1<f64> = leakages
                     .row(row)
                     .slice(s![start_sample..end_sample])
                     .map(|l| *l as f64);
                 let sample_metadat: Array1<usize> = plaintext.row(row).map(|p| *p as usize);
                 cpa.update(sample_trace, sample_metadat);
             }
             cpa
         })
         .reduce(|| Cpa::new(size, guess_range, leakage_model), |x, y| x + y);
 
     cpa_parallel.finalize();
     println!("Guessed key = {}", cpa_parallel.pass_guess());
     // save corr key curves in npy
     write_array("../results/corr.npy", cpa_parallel.pass_corr_array().view());
 }
 

fn success_rate() {
    let start_sample: usize = 0;
    let end_sample: usize = 5000;
    let size: usize = end_sample - start_sample; // Number of samples
    let guess_range = 256; // 2**(key length)
    let folder = String::from("../data/old"); // Directory of leakages and metadata
    let nfiles: i32 = 5; // Number of files in the directory. TBD: Automating this value
    let success_no = 1000;
    let mut cpa = Cpa::new(size, guess_range, leakage_model);
    for n_files in (0..nfiles).progress() {
        let dir_l = format!("{folder}/l{n_files}.npy");
        let dir_p = format!("{folder}/p{n_files}.npy");
        let leakages: Array2<FormatTraces> = read_array_2_from_npy_file::<FormatTraces>(&dir_l);
        let plaintext: Array2<FormatMetadata> =
            read_array_2_from_npy_file::<FormatMetadata>(&dir_p);
        let no_traces = leakages.shape()[0];
        // let mut cpa_inner = Cpa::new(size, guess_range, leakage_model);
        for block in (0..no_traces).step_by(success_no) {
            let l_chunk: ArrayView2<FormatTraces> =
                leakages.slice(s![block..block + success_no, start_sample..end_sample]);
            let p_chunk: ArrayView2<FormatMetadata> =
                plaintext.slice(s![block..block + success_no, ..]);
            let cpa_inner = (0..success_no)
                .into_par_iter()
                .map(|index| {
                    let mut c = Cpa::new(size, guess_range, leakage_model);
                    c.update(
                        l_chunk.row(index).map(|l| *l as f64),
                        p_chunk.row(index).to_owned(),
                    );
                    c
                })
                .reduce(|| Cpa::new(size, guess_range, leakage_model), |x, y| x + y);

            cpa = cpa + cpa_inner;
            cpa.finalize();
            cpa.update_success();
        }

        write_array("../results/success_rate.npy", cpa.pass_succes().view());
    }

}




fn main() {
    let mut t = time::Instant::now();
    cpa_cw();
    println!("{:?}", t.elapsed());
}

