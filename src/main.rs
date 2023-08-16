use std:: error;
use ndarray::{Array2, s, ArrayView1};
/* import other files */
mod leakage;
use leakage:: leakage_model;
mod read;
use read:: *;
use simple_bar::ProgressBar;



#[derive(Default)]
pub struct Cpa{
    sumleakages: Vec<usize>,
    sigleakages: Vec<usize>,
    sumkeys: Vec<usize>,
    sigkeys: Vec<usize>,
    values: Vec<FormatMetadata>,
    _al: Array2<usize>,
    start: bool,
    _target_byte: i32,
    len_leakages: usize,
    _guess_range: i32,
    corr: Array2<f32>,
    len_samples: usize
}


impl Cpa {
    fn update(&mut self, trace: Vec<FormatTraces>, plaintext: Vec<FormatMetadata>
        , target_key: i32, guess_range: i32){
        
        /* This function updates the main arrays of the CPA, as shown in Alg. 4
        in the paper.*/

        if ! self.start{
            self.len_leakages = 0;
            self.len_samples = trace.len();
            self._al = Array2::zeros((guess_range as usize, self.len_samples));
            self._target_byte = target_key;
            self._guess_range = guess_range;
            self.sumleakages =  vec! [0; self.len_samples];
            self.sigleakages =  vec! [0; self.len_samples];
            self.sumkeys =  vec! [0; guess_range as usize];
            self.sigkeys =  vec! [0; guess_range as usize];
            self.values = vec![0; guess_range as usize]; 
            self.start = true;
            self.corr =  Array2::zeros((guess_range as usize, self.len_samples));
        }

        self.len_leakages += 1;
        self.gen_values(plaintext.clone(), guess_range, target_key);
        self.go(trace, plaintext.clone(), guess_range);   
    }

    fn gen_values(&mut self, metadata: Vec<FormatMetadata>,
         _guess_range: i32, _target_key: i32 ){
        
        for guess in 0.._guess_range{
            self.values[guess as usize] = leakage_model(metadata[_target_key as usize], guess);
        }  
    }


    fn go(&mut self, _trace:Vec<FormatTraces>, metadata: Vec<FormatMetadata>, _guess_range: i32){
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

    fn finalize(&mut self){
        /* This function finalizes the calculation after feeding the
        overall traces */
        
        let shape_p = self._guess_range as usize;
        let mut p: ndarray::ArrayBase<ndarray::OwnedRepr<usize>, ndarray::Dim<[usize; 2]>> = Array2::zeros((shape_p , shape_p));
        for i in 0..self._guess_range{
            for x in 0..self._guess_range{
                p[[x as usize, i as usize]] = leakage_model(x, i) as usize;
            }
        } 
   
        for i in 0..self._guess_range{
            let _sigkeys = self.sigkeys[i as usize] as f32 / self.len_leakages as f32;
            let _sumkeys = self.sumkeys[i as usize] as f32 / self.len_leakages as f32;
            let lower1: f32 = _sigkeys - (_sumkeys * _sumkeys); 
            
            for x in 0..self.len_samples{
                let _sumleakages = self.sumleakages[x as usize] as f32 / self.len_leakages as f32;
                let _sigleakages = self.sigleakages[x as usize] as f32 / self.len_leakages as f32;
                let slice_a = self._al.slice(s![.., x]);
                let slice_b = p.slice(s![.., i]);
                let summult: i32 = self.sum_mult(slice_a, slice_b);
                let upper1: f32 = summult as f32 / self.len_leakages as f32;                
                let upper: f32 = upper1 - ((_sumkeys * _sumleakages));
                let lower2: f32 = _sigleakages - (_sumleakages * _sumleakages);
                let lower = f32::sqrt(lower1 * lower2);
                self.corr[[i as usize, x]] = f32::abs(upper / lower);                
            }
        }
        self.find_guess();

    }


    fn find_guess(&self){
        let mut max_256: Vec<f32> = vec![0.0; 256];
        for i in 0..self._guess_range{
            let row = self.corr.row(i as usize);
            max_256[i as usize] = *row.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        }
        let mut init_value: f32 = 0.0;
        let mut guess: i32 = 0;
        for i in 0..self._guess_range{
            if max_256[i as usize] > init_value{
                init_value = max_256[i as usize];
                guess = i;
            }
        }
        println!("guessed key = {}", guess);
        let _ = write_npy("corr.npy", self.corr.clone());
        println!("Result is saved in .npy!");
    }

    fn sum_mult(&self, a: ArrayView1<usize>, b: ArrayView1<usize>) -> i32 {
        a.dot(&b) as i32
    }

}



fn main()-> Result<(), Box<dyn error::Error>>{
    let mut c: Cpa = Default::default();
    let dir_leakages: &str  = "data/leakages.npy"; 
    let dir_metadat: &str = "data/plaintext.npy";
    let leakages: ndarray::ArrayBase<ndarray::OwnedRepr<FormatTraces>, ndarray::Dim<[usize; 2]>> = read_leakages(dir_leakages)?;
    let plaintext: ndarray::ArrayBase<ndarray::OwnedRepr<FormatMetadata>, ndarray::Dim<[usize; 2]>> = read_metadata(dir_metadat)?;
    let len_leakages = leakages.shape()[0];
    let num_iter = len_leakages as u32;
    let mut bar = ProgressBar::default(num_iter, 50, false);
    for i in 0..len_leakages{
        c.update(leakages.row(i).to_vec(), plaintext.row(i).to_vec(), 1, 256);
        bar.update();
    }
    c.finalize();
    Ok(())
}