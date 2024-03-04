use cpa::cpa_single::*;
use cpa::leakage::{hw, sbox};
use cpa::tools::{read_array_2_from_npy_file, write_array};
use indicatif::ProgressIterator;
use ndarray::*;
use std::time::{self};

// traces format
type FormatTraces = i16;
type FormatMetadata = i32;

// leakage model
pub fn leakage_model(value: Array1<FormatMetadata>, guess: usize) -> f64 {
    hw(sbox(value[1] as u8 ^ guess as u8) as usize) as f64
}

fn cpa() {
    let start_sample: usize = 0;
    let end_sample: usize = 1000;
    let size: usize = end_sample - start_sample; // Number of samples
    let guess_range: i32 = 256; // 2**(key length)
    let folder = String::from("data/old"); // Directory of leakages and metadata
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

fn main() {
    let t = time::Instant::now();
    cpa();
    println!("{:?}", t.elapsed());
}



