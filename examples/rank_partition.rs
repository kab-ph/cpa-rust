use cpa::cpa_partition::*;
use cpa::leakage::{hw, sbox};
use cpa::tools::{read_array_2_from_npy_file, write_array};
use ndarray::*;
use rayon::prelude::{ParallelBridge, ParallelIterator};
use simple_bar::ProgressBar;
use std::time::Instant;

// traces format
type FormatTraces = i16;
type FormatMetadata = i32;

// leakage model
pub fn leakage_model(value: usize, guess: usize) -> usize {
    hw(sbox((value ^ guess) as u8) as usize)
}

fn rank() {
    let size: usize = 5000; // Number of samples
    let guess_range = 256; // 2**(key length)
    let target_byte = 1;
    let folder = String::from("../data/old");
    let nfiles = 5;
    let mut bar = ProgressBar::default(nfiles as u32, 50, false);
    let chunk = 3000;
    let mut rank = Cpa_partition::new(size, guess_range, target_byte, leakage_model);
    for file in 0..nfiles {
        let dir_l = format!("{folder}/l{file}.npy"); // leakage directory
        let dir_p = format!("{folder}/p{file}.npy"); // plaintext directory
        let leakages: Array2<FormatTraces> = read_array_2_from_npy_file::<FormatTraces>(&dir_l);
        let plaintext: Array2<FormatMetadata> =
            read_array_2_from_npy_file::<FormatMetadata>(&dir_p);
        let len_file = leakages.shape()[0];
        for sample in (0..len_file).step_by(chunk) {
            let l_sample: ndarray::ArrayBase<
                ndarray::ViewRepr<&FormatTraces>,
                ndarray::Dim<[usize; 2]>,
            > = leakages.slice(s![sample..sample + chunk, ..]);
            let p_sample = plaintext.slice(s![sample..sample + chunk, ..]);
            let x = (0..chunk)
                .into_iter()
                .par_bridge()
                .fold(
                    || Cpa_partition::new(size, guess_range, target_byte, leakage_model),
                    |mut r: Cpa_partition, n| {
                        r.update(
                            l_sample.row(n).map(|l: &FormatTraces| *l as usize),
                            p_sample.row(n).map(|p: &FormatMetadata| *p as usize),
                        );
                        r
                    },
                )
                .reduce(
                    || Cpa_partition::new(size, guess_range, target_byte, leakage_model),
                    |lhs, rhs| lhs + rhs,
                );
            rank = rank + x;
            rank.finalize();
        }
        bar.update();
    }
    // save rank key curves in npy
    write_array("../results/rank.npy", rank.pass_rank());
}

fn main() {
    let t = Instant::now();
    rank();
    println!("Time for key rank {:?}", t.elapsed());
}
