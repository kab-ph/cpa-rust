use ndarray::{Array2, s, ArrayView1, ArrayView2, Array1, concatenate, Axis};
use rayon::prelude::{ParallelIterator, IntoParallelIterator};
use std::ops::Add;
// use crate::leakage::leakage_model;
/* import other files */


pub struct Cpa{
    sumleakages: Array1<usize>,
    sigleakages: Array1<usize>,
    sumkeys: Array1<usize>,
    sigkeys: Array1<usize>,
    values: Array1<usize>,
    _al: Array2<usize>,
    _target_byte: i32,
    len_leakages: usize,
    _guess_range: i32,
    corr: Array2<f32>,
    rank_slice: Array2<f32>,
    leakage_func: fn(usize, usize) -> usize,
    len_samples: usize
}


impl Cpa {
    pub fn new(size: usize, guess_range: i32, target_byte: i32, f: fn(usize, usize)->usize) -> Self{
        Self{
            len_samples: size,
            _al: Array2::zeros((guess_range as usize, size)),
            _target_byte: target_byte,
            _guess_range: guess_range,
            sumleakages: Array1::zeros(size),
            sigleakages: Array1::zeros(size),
            sumkeys: Array1::zeros(guess_range as usize),
            sigkeys: Array1::zeros(guess_range as usize),
            values: Array1::zeros(guess_range as usize),
            corr:  Array2::zeros((guess_range as usize, size)),
            rank_slice: Array2::zeros((guess_range as usize, 1)),
            leakage_func: f,
            len_leakages:0,
        }
    }
    
    pub fn update(&mut self, trace: Array1<usize>, plaintext: Array1<usize>){
        
        /* This function updates the main arrays of the CPA, as shown in Alg. 4
        in the paper.*/
        self.len_leakages += 1;
        self.gen_values(plaintext.clone(), self._guess_range, self._target_byte);
        self.go(trace, plaintext.clone(), self._guess_range);   
    }

    pub fn gen_values(&mut self, metadata: Array1<usize>,
         _guess_range: i32, _target_key: i32 ){
        
        for guess in 0.._guess_range{
            self.values[guess as usize] = (self.leakage_func)(metadata[_target_key as usize], guess as usize) as usize;
        }  
    }


    pub fn go(&mut self, _trace:Array1<usize>, metadata: Array1<usize>, _guess_range: i32){
        for i in 0.. self.len_samples{
            self.sumleakages[i] += _trace[i] as usize;
            self.sigleakages[i] += (_trace[i] * _trace[i]) as usize;
        }

        for guess in 0.._guess_range{
            self.sumkeys[guess as usize] += self.values[guess as usize] as usize;
            self.sigkeys[guess as usize] += (self.values[guess as usize] * self.values[guess as usize]) as usize;
        }
        let partition: usize = metadata[self._target_byte as usize] as usize;
        for i in 0..self.len_samples{
            self._al[[partition, i]] += _trace[i] as usize;
        } 
    }

    pub fn finalize(&mut self) -> Array2<f32>{
        /* This function finalizes the calculation after feeding the
        overall traces */
        
        let shape_p = self._guess_range as usize;
        let mut p: ndarray::ArrayBase<ndarray::OwnedRepr<usize>, ndarray::Dim<[usize; 2]>> = Array2::zeros((shape_p , shape_p));
        for i in 0..self._guess_range{
            for x in 0..self._guess_range{
                p[[x as usize, i as usize]] = ((self.leakage_func))(x as usize, i as usize) as usize;
            }
        } 


        
        for i in 0..self._guess_range{
            let _sigkeys = self.sigkeys[i as usize] as f32 / self.len_leakages as f32;
            let _sumkeys = self.sumkeys[i as usize] as f32 / self.len_leakages as f32;
            let lower1: f32 = _sigkeys - (_sumkeys * _sumkeys); 
            /* Parallel operation using multi-threading */
            let tmp: Vec<f32> = (0..self.len_samples).into_par_iter().
            map(|x|
            {
                let _sumleakages = self.sumleakages[x as usize] as f32 / self.len_leakages as f32;
                let _sigleakages = self.sigleakages[x as usize] as f32 / self.len_leakages as f32;
                let slice_a = self._al.slice(s![.., x]);
                let slice_b = p.slice(s![.., i]);
                let summult: i32 = self.sum_mult(slice_a, slice_b);
                let upper1: f32 = summult as f32 / self.len_leakages as f32;                
                let upper: f32 = upper1 - ((_sumkeys * _sumleakages));
                let lower2: f32 = _sigleakages - (_sumleakages * _sumleakages);
                let lower = f32::sqrt(lower1 * lower2);
                f32::abs(upper / lower)               
            }).collect();

            for z in 0..self.len_samples{
                self.corr[[i as usize, z]] = tmp[z];

            }
        }
        self.find_guess();
        self.corr.clone()

    }

    pub fn find_guess(&mut self){
        let mut max_256: Array2<f32> = Array2::zeros((self._guess_range as usize, 1));
        for i in 0..self._guess_range{
            let row = self.corr.row(i as usize);
            max_256[[i as usize, 0]] = *row.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        }
        self.rank_slice = concatenate![Axis(1), self.rank_slice, max_256];
        let mut init_value: f32 = 0.0;
        let mut guess: i32 = 0;
        for i in 0..self._guess_range{
            if max_256[[i as usize, 0]] > init_value{
                init_value = max_256[[i as usize, 0]];
                guess = i;
            }
        }
        // println!("guessed key = {}", guess);
        // let _ = write_npy("corr.npy", self.corr.clone().view());
        // println!("Result is saved in .npy!");
    }
    
    pub fn pass_rank(&self) -> ArrayView2<f32>{
        self.rank_slice.slice(s![.., 1..])
    }


    fn sum_mult(&self, a: ArrayView1<usize>, b: ArrayView1<usize>) -> i32 {
        a.dot(&b) as i32
    }


}


impl Add for Cpa {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            sumleakages: self.sumleakages + rhs.sumleakages,
            sigleakages: self.sigleakages + rhs.sigleakages,
            sumkeys: self.sumkeys + rhs.sumkeys,
            sigkeys: self.sigkeys + rhs.sigkeys,
            values: self.values + rhs.values,
            _al: self._al + rhs._al,
            _target_byte: rhs._target_byte,
            len_leakages: self.len_leakages + rhs.len_leakages,
            _guess_range: rhs._guess_range,
            corr: self.corr + rhs.corr,
            rank_slice: self.rank_slice, 
            len_samples: rhs.len_samples,
            leakage_func: self.leakage_func
        }
    }
}













