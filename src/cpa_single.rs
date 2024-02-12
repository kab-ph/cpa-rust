use ndarray::{concatenate, s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::ops::Add;

pub struct Cpa {
    /* List of internal class variables */
    sum_leakages: Array1<usize>,
    sig_leakages: Array1<usize>,
    sum_keys: Array1<usize>,
    sig_keys: Array1<usize>,
    values: Array1<usize>,
    len_leakages: usize,
    guess_range: i32,
    cov: Array2<usize>,
    corr: Array2<f32>,
    max_corr: Array2<f32>,
    rank_slice: Array2<f32>,
    leakage_func: fn(ArrayView1<usize>, usize) -> usize,
    len_samples: usize,
    rank_traces: usize, // Number of traces to calculate succes rate
}

/* This class implements the CPA algorithm shown in:
https://www.iacr.org/archive/ches2004/31560016/31560016.pdf */

impl Cpa {
    pub fn new(
        size: usize,
        guess_range: i32,
        f: fn(ArrayView1<usize>, usize) -> usize,
    ) -> Self {
        Self {
            len_samples: size,
            guess_range: guess_range,
            sum_leakages: Array1::zeros(size),
            sig_leakages: Array1::zeros(size),
            sum_keys: Array1::zeros(guess_range as usize),
            sig_keys: Array1::zeros(guess_range as usize),
            values: Array1::zeros(guess_range as usize),
            cov: Array2::zeros((guess_range as usize, size)),
            corr: Array2::zeros((guess_range as usize, size)),
            max_corr: Array2::zeros((guess_range as usize, 1)),
            rank_slice: Array2::zeros((guess_range as usize, 1)),
            leakage_func: f,
            len_leakages: 0,
            rank_traces: 0,
            // traces_patch: Array2::zeros((patch, size)),
        }
    }

    pub fn update(&mut self, trace: Array1<usize>, metadata: Array1<usize>) {
        
        self.update_1(&metadata);
        self.update_key_leakages(&trace);
        self.update_cov(&trace);
        self.len_leakages += 1;
    }

    pub fn update_1(&mut self, metadata: &Array1<usize>){
        for guess in 0..self.guess_range {
            self.values[guess as usize] =
                (self.leakage_func)(metadata.view(), guess as usize);
        }

    }



    pub fn update_cov(
        /* This function generates the values and cov arrays */
        &mut self, sample_trace: &Array1<usize>
        
    ) {
        
        
        /* Parallelism is used to update the cov array */
        for column in 0..self.len_samples {
            for row in 0..self.guess_range{
                self.cov[[row as usize, column]] += self.values[row as usize] * sample_trace[column]; 

            }
        }
    }

    pub fn update_key_leakages(&mut self, sample_trace: &Array1<usize>) {
        for i in 0..self.len_samples {
            self.sum_leakages[i] += sample_trace[i]; 
            self.sig_leakages[i] += sample_trace[i] * sample_trace[i]; 
        }

        for guess in 0..self.guess_range {
            self.sum_keys[guess as usize] += self.values[guess as usize] as usize;
            self.sig_keys[guess as usize] += (self.values[guess as usize] * self.values[guess as usize]) as usize;
        }
    }


    pub fn finalize(&mut self) {
        // println!("{:?}", self.cov);
        /* This function finalizes the calculation after feeding the
        overall traces */

        for i in 0..self.guess_range as i32 {
            for x in 0..self.len_samples {
                let upper: f32 = (self.cov[[i as usize, x]] as f32 / self.len_leakages as f32)
                    - ((self.sum_keys[i as usize] as f32 / self.len_leakages as f32)
                        * (self.sum_leakages[x] as f32 / self.len_leakages as f32));

                let lower_1 = (self.sig_keys[i as usize] as f32 / self.len_leakages as f32)
                    - ((self.sum_keys[i as usize] as f32 / self.len_leakages as f32)
                        * (self.sum_keys[i as usize] as f32 / self.len_leakages as f32));

                let lower_2 = (self.sig_leakages[x as usize] as f32 / self.len_leakages as f32)
                    - ((self.sum_leakages[x as usize] as f32 / self.len_leakages as f32)
                        * (self.sum_leakages[x as usize] as f32 / self.len_leakages as f32));

                self.corr[[i as usize, x]] = f32::abs(upper / f32::sqrt(lower_1 * lower_2));
            }
        }
        self.calculation();
    }

    pub fn calculation(&mut self) {
        for i in 0..self.guess_range {
            let row = self.corr.row(i as usize);
            // Calculating the max value in the row
            let max_value = row
                .into_iter()
                .reduce(|a, b| {
                    let mut tmp = a;
                    if tmp < b {
                        tmp = b;
                    }
                    tmp
                })
                .unwrap();
            self.max_corr[[i as usize, 0]] = *max_value;
        }
    }

    pub fn success_traces(&mut self, traces_no: usize) {
        self.rank_traces = traces_no;
    }

    pub fn pass_rank(&self) -> ArrayView2<f32> {
        self.rank_slice.slice(s![.., 1..])
    }

    pub fn pass_corr_array(&self) -> Array2<f32> {
        self.corr.clone()
    }

    pub fn pass_guess(&self) -> i32 {
        let mut init_value: f32 = 0.0;
        let mut guess: i32 = 0;
        for i in 0..self.guess_range {
            if self.max_corr[[i as usize, 0]] > init_value {
                init_value = self.max_corr[[i as usize, 0]];
                guess = i;
            }
        }

        guess
    }
}


impl Add for Cpa{
    type Output = Self;
    fn add(self, rhs: Self) -> Self{
        Self { sum_leakages: self.sum_leakages + rhs.sum_leakages,
             sig_leakages: self.sig_leakages + rhs.sig_leakages, 
             sum_keys: self.sum_keys + rhs.sum_keys, 
             sig_keys: self.sig_keys + rhs.sig_keys,
             values: self.values,
             len_leakages: self.len_leakages + rhs.len_leakages, 
             guess_range: self.guess_range, 
             cov: self.cov + rhs.cov, 
             corr: self.corr, 
             max_corr: self.max_corr, 
             rank_slice: self.rank_slice, 
             leakage_func: self.leakage_func, 
             len_samples: self.len_samples, 
             rank_traces: self.rank_traces }

    }
}