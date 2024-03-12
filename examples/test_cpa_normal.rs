use cpa::cpa_normal::*;
use cpa::leakage::{hw, sbox};
use cpa::tools::{read_array_2_from_npy_file, write_array};
use indicatif::ProgressIterator;
use ndarray::*;
use std::time::Instant;

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
    let patch: usize = 500;
    let guess_range = 256; // 2**(key length)
    let folder = String::from("data/old"); // Directory of leakages and metadata
    let nfiles = 5; // Number of files in the directory. TBD: Automating this value
    let rank_traces = 2000;
    let mut cpa = Cpa::new(size, patch, guess_range, leakage_model);
    cpa.success_traces(rank_traces);
    for i in (0..nfiles).progress() {
        let dir_l = format!("{folder}/l{i}.npy");
        let dir_p = format!("{folder}/p{i}.npy");
        let leakages: Array2<FormatTraces> = read_array_2_from_npy_file::<FormatTraces>(&dir_l);
        let plaintext: Array2<FormatMetadata> =
            read_array_2_from_npy_file::<FormatMetadata>(&dir_p);
        let len_leakages = leakages.shape()[0];
        for row in (0..len_leakages).step_by(patch) {
            let range_samples = start_sample..end_sample;
            let range_rows = row..row + patch;
            let range_metadat = 0..plaintext.shape()[1];
            let sample_traces = leakages
                .slice(s![range_rows.clone(), range_samples])
                .map(|l| *l as f32);
            let sample_metadata = plaintext
                .slice(s![range_rows, range_metadat])
                .map(|p| *p as usize);
            cpa.update(sample_traces, sample_metadata);
        }
    }

    cpa.finalize();
    println!("Guessed key = {}", cpa.pass_guess());
    // save corr key curves in npy
    write_array("../results/corr.npy", cpa.pass_corr_array().view());
}

fn main() {
    let t = Instant::now();
    cpa();
    println!("{:?}", t.elapsed());
}
