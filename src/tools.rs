use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array2, ArrayView2};
use ndarray_npy::{ReadNpyExt, ReadableElement, WriteNpyExt};
use std::io::BufWriter;
use std::{fs::File, time::Duration};

use plotly::common::Title;
use plotly::{Plot, Scatter};

pub fn read_array_2_from_npy_file<T: ReadableElement>(dir: &str) -> Array2<T> {
    let reader: File = File::open(dir).unwrap();
    let arr: Array2<T> = Array2::<T>::read_npy(reader).unwrap();
    arr
}

pub fn write_array(dir: &str, ar: ArrayView2<f32>) {
    let writer = BufWriter::new(File::create(dir).unwrap());
    ar.write_npy(writer).unwrap();
}

/// Creates a [`ProgressBar`] with a predefined default style.
pub fn progress_bar(len: usize) -> ProgressBar {
    let progress_bar = ProgressBar::new(len as u64).with_style(
        ProgressStyle::with_template("{elapsed_precise} {wide_bar} {pos}/{len} ({eta})").unwrap(),
    );
    progress_bar.enable_steady_tick(Duration::new(0, 100000000));
    progress_bar
}

pub fn plot_array2(arr: Array2<f32>, name: String, t: String) -> Plot {
    let mut plot: Plot = Plot::new();
    let x: Vec<f32> = (0..arr.shape()[1]).into_iter().map(|x| x as f32).collect();
    for i in 0..arr.shape()[0] {
        let trace = arr.row(i).to_vec();
        // let trace = Scatter::new(x.clone(), trace).name(format!("{name} [{i}]"));
        let trace = Scatter::new(x.clone(), trace).name(format!("{name}[{i}]"));
        plot.add_trace(trace);
    }
    let title: &str = &t;
    plot.set_layout(plotly::Layout::new().title(Title::from(title)));
    plot
}
