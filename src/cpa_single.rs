use ndarray::{concatenate, s, Array1, Array2, ArrayView2, Axis};
use std::ops::Add;

pub struct Cpa<T> {
    /* List of internal class variables */
    sum_leakages: Array1<f64>,
    sig_leakages: Array1<f64>,
    sum_keys: Array1<f64>,
    sig_keys: Array1<f64>,
    values: Array1<f64>,
    len_leakages: usize,
    guess_range: i32,
    cov: Array2<f64>,
    corr: Array2<f32>,
    max_corr: Array2<f32>,
    rank_slice: Array2<f32>,
    init_rank: bool,
    leakage_func: fn(T, usize) -> f64,
    len_samples: usize,
    rank_traces: usize, // Number of traces to calculate succes rate
}

/* This class implements the CPA algorithm shown in:
https://www.iacr.org/archive/ches2004/31560016/31560016.pdf */

impl<T: Clone> Cpa<T> {
    pub fn new(size: usize, guess_range: i32, f: fn(T, usize) -> f64) -> Self {
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
            init_rank: false, // traces_patch: Array2::zeros((patch, size)),
        }
    }

    pub fn update<U: Clone>(&mut self, trace: Array1<U>, metadata: T)
    where
        f64: From<U>,
    {
        let mut trace_tmp: Array1<f64> = Array1::zeros(self.len_samples);
        for i in 0..self.len_samples as usize {
            trace_tmp[i] = trace[i].clone().into();
        }

        self.update_values(&metadata);
        self.update_arrays(&trace_tmp);
        self.update_cov(&trace_tmp);
        self.len_leakages += 1;
    }

    pub fn update_values(&mut self, metadata: &T) {
        for guess in 0..self.guess_range {
            self.values[guess as usize] = (self.leakage_func)(metadata.clone(), guess as usize);
        }
    }

    pub fn update_cov(
        /* This function generates the values and cov arrays */
        &mut self,
        sample_trace: &Array1<f64>,
    ) {
        /* Update cov */
        for column in 0..self.len_samples {
            for row in 0..self.guess_range {
                self.cov[[row as usize, column]] +=
                    self.values[row as usize] as f64 * sample_trace[column];
            }
        }

        // let mut tmp_values: Array2<f64> = Array2::zeros((self.guess_range as usize, 1));
        // tmp_values.column_mut(0).assign(&self.values.map(|x| *x as f64));
        // let tmp_cov: Array2<f64> = tmp_values * sample_trace;
        // self.cov = self.cov.clone() + tmp_cov;

        // let mut tmp_cov: Array2<f64> = Array2::zeros((self.guess_range as usize, self.len_samples));
        // for column in 0..self.len_samples{
        //     let tmp_column = sample_trace[column] * self.values.map(|x| *x as f64);
        //     tmp_cov.column_mut(column).assign(&tmp_column);
        // }
        // self.cov = tmp_cov; //self.cov.clone() + tmp_cov;

        // let mut tmp_cov: Array2<f64> = Array2::zeros((self.guess_range as usize, self.len_samples));
        // for row in 0..self.guess_range as usize{
        //     let tmp_row = sample_trace * self.values[row] as f64;
        //     tmp_cov.row_mut(row).assign(&tmp_row);
        // }
        // self.cov = self.cov.clone() + tmp_cov;
    }

    pub fn update_arrays(&mut self, sample_trace: &Array1<f64>) {
        for i in 0..self.len_samples {
            self.sum_leakages[i] += sample_trace[i];
            self.sig_leakages[i] += sample_trace[i] * sample_trace[i];
        }

        for guess in 0..self.guess_range {
            self.sum_keys[guess as usize] += self.values[guess as usize];
            self.sig_keys[guess as usize] +=
                self.values[guess as usize] * self.values[guess as usize];
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

    pub fn update_success(&mut self) {
        if !self.init_rank {
            self.rank_slice = self.max_corr.clone();
            self.init_rank = true;
        } else {
            self.rank_slice = concatenate![Axis(1), self.rank_slice.clone(), self.max_corr];
        }
    }

    pub fn pass_succes(&self) -> Array2<f32> {
        self.rank_slice.clone()
    }
}

impl<T> Add for Cpa<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            sum_leakages: self.sum_leakages + rhs.sum_leakages,
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
            rank_traces: self.rank_traces,
            init_rank: self.init_rank,
        }
    }
}
