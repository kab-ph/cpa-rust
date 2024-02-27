use ndarray::{concatenate, s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::ops::Add;
pub struct Cpa {
    /* List of internal class variables */
    sum_leakages: Array1<usize>,
    sig_leakages: Array1<usize>,
    sum_keys: Array1<usize>,
    sig_keys: Array1<usize>,
    values: Array2<usize>,
    len_leakages: usize,
    guess_range: i32,
    cov: Array2<usize>,
    corr: Array2<f32>,
    max_corr: Array2<f32>,
    rank_slice: Array2<f32>,
    leakage_func: fn(ArrayView1<usize>, usize) -> usize,
    len_samples: usize,
    chunk: usize,
    rank_traces: usize, // Number of traces to calculate succes rate
}

/* This class implements the CPA algorithm shown in:
https://www.iacr.org/archive/ches2004/31560016/31560016.pdf */

impl Cpa {
    pub fn new(
        size: usize,
        patch: usize,
        guess_range: i32,
        f: fn(ArrayView1<usize>, usize) -> usize,
    ) -> Self {
        Self {
            len_samples: size,
            chunk: patch,
            guess_range: guess_range,
            sum_leakages: Array1::zeros(size),
            sig_leakages: Array1::zeros(size),
            sum_keys: Array1::zeros(guess_range as usize),
            sig_keys: Array1::zeros(guess_range as usize),
            values: Array2::zeros((patch, guess_range as usize)),
            cov: Array2::zeros((guess_range as usize, size)),
            corr: Array2::zeros((guess_range as usize, size)),
            max_corr: Array2::zeros((guess_range as usize, 1)),
            rank_slice: Array2::zeros((guess_range as usize, 1)),
            leakage_func: f,
            len_leakages: 0,
            rank_traces: 0,
        }
    }

    pub fn update(&mut self, trace_patch: Array2<usize>, plaintext_patch: Array2<usize>) {
        /* This function updates the internal arrays of the CPA
        It accepts trace_patch and plaintext_patch to update them*/

        self.len_leakages += self.chunk;
        self.update_values(&plaintext_patch, &trace_patch, self.guess_range);
        self.update_key_leakages(trace_patch, self.guess_range);
    }

    pub fn update_values(
        /* This function generates the values and cov arrays */
        &mut self,
        metadata: &Array2<usize>,
        _trace: &Array2<usize>,
        _guess_range: i32,
    ) {
        for row in 0..self.chunk {
            for guess in 0.._guess_range {
                let pass_to_leakage: ArrayView1<usize> = metadata.row(row);
                self.values[[row, guess as usize]] =
                    (self.leakage_func)(pass_to_leakage, guess as usize) as usize;
            }
        }

        for column in 0..self.len_samples {
            for row in 0..self.guess_range {
                self.cov[[row as usize, column]] += self
                    .values
                    .column(row as usize)
                    .dot(&_trace.column(column as usize));
            }
        }
        /* Parallelism is used to update the cov array */
        // for row in 0..self.guess_range {
        //     let row_cov: Vec<usize> = (0..self.len_samples)
        //         .into_par_iter()
        //         .map(|index_column| {
        //             _trace
        //                 .column(index_column)
        //                 .dot(&self.values.column(row as usize))
        //         })
        //         .collect();
        //
        //     let mut r: Array1<usize> = row_cov.into();
        //     r = r.clone() + self.cov.row(row as usize);
        //     self.cov.row_mut(row as usize).assign(&r);
        // }
    }

    pub fn update_key_leakages(&mut self, _trace: Array2<usize>, _guess_range: i32) {
        for i in 0..self.len_samples {
            self.sum_leakages[i] += _trace.column(i).sum(); // _trace[i] as usize;
            self.sig_leakages[i] += _trace.column(i).dot(&_trace.column(i)); // (_trace[i] * _trace[i]) as usize;
        }

        for guess in 0.._guess_range {
            self.sum_keys[guess as usize] += self.values.column(guess as usize).sum(); //self.values[guess as usize] as usize;
            self.sig_keys[guess as usize] += self
                .values
                .column(guess as usize)
                .dot(&self.values.column(guess as usize));
            // (self.values[guess as usize] * self.values[guess as usize]) as usize;
        }
    }

    pub fn update_success(&mut self, trace_patch: Array2<usize>, plaintext_patch: Array2<usize>) {
        /* This function updates the main arrays of the CPA for the success rate*/
        self.update(trace_patch, plaintext_patch);
        if self.len_leakages % self.rank_traces == 0 {
            self.finalize();
            if self.len_leakages == self.rank_traces {
                self.rank_slice = self.max_corr.clone();
            } else {
                self.rank_slice = concatenate![Axis(1), self.rank_slice, self.max_corr];
            }
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

impl Add for Cpa {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            sum_leakages: self.sum_leakages + rhs.sum_leakages,
            sig_leakages: self.sig_leakages + rhs.sig_leakages,
            sum_keys: self.sum_keys + rhs.sum_keys,
            sig_keys: self.sig_keys + rhs.sig_keys,
            values: self.values + rhs.values,
            len_leakages: self.len_leakages + rhs.len_leakages,
            guess_range: rhs.guess_range,
            chunk: rhs.chunk,
            cov: self.cov + rhs.cov,
            corr: self.corr + rhs.corr,
            max_corr: self.max_corr,
            rank_slice: self.rank_slice,
            len_samples: rhs.len_samples,
            leakage_func: self.leakage_func,
            rank_traces: self.rank_traces,
        }
    }
}
