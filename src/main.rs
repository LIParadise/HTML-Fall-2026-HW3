use faer::{
    Scale,
    col::{Col, ColRef},
    mat::{Mat, MatRef},
};
use std::{fs::File, io::read_to_string, path::Path};

const DIM: usize = 10;

#[derive(Clone, Copy, Debug)]
enum Label {
    Positive,
    Negative,
}
impl From<f64> for Label {
    fn from(value: f64) -> Self {
        match value.is_sign_positive() {
            true => Self::Positive,
            _ => Self::Negative,
        }
    }
}
impl From<Label> for f64 {
    fn from(value: Label) -> Self {
        match value {
            Label::Positive => 1.0,
            Label::Negative => -1.0,
        }
    }
}

#[derive(Clone, Debug)]
struct LabeledPoint {
    coordinates: Vec<f64>,
    label: Label,
}
impl LabeledPoint {
    /// Add dimension in the _front_
    /// TODO: without relocate?
    fn augment(self, v: f64) -> Self {
        Self {
            coordinates: Vec::from_iter(std::iter::once(v).chain(self.coordinates)),
            ..self
        }
    }
}

fn get_data<const N: usize, P: AsRef<Path>>(p: P) -> Vec<LabeledPoint> {
    read_to_string(File::open(p).unwrap())
        .unwrap()
        .lines()
        .map(|s| {
            let iter = &mut s
                .split_whitespace()
                .map(str::parse::<f64>)
                .map(Result::unwrap);

            let coordinates = iter.take(N).collect::<Vec<_>>();
            assert_eq!(
                coordinates.len(),
                N,
                "robust against missing data is a TODO"
            );
            let label = iter.map(Label::from).next().unwrap();
            LabeledPoint { coordinates, label }
        })
        .collect()
}

/// TODO: without relocate?
fn augment_data(data: Vec<LabeledPoint>, v: f64) -> Vec<LabeledPoint> {
    data.into_iter().map(|p| p.augment(v)).collect()
}

fn pw_sgd(l: Label, w: ColRef<f64>, x: ColRef<f64>) -> f64 {
    sigmoid(-f64::from(l) * (w.transpose() * x))
}
fn sigmoid(f: f64) -> f64 {
    1.0 / (1.0 + (-f).exp())
}

fn q11() -> (f64, Col<f64>) {
    let data = augment_data(get_data::<DIM, _>("./hw3_train.txt"), 1.0);
    let mut mat = Mat::<f64, _, _>::zeros(data.len(), DIM + 1);
    mat.row_iter_mut()
        .zip(&data)
        .for_each(|(row, LabeledPoint { coordinates, .. })| {
            row.iter_mut()
                .zip(coordinates)
                .for_each(|(coordinate, c)| *coordinate = *c);
        });
    let svd = faer::linalg::solvers::Svd::new(mat.as_ref()).unwrap();
    let pinv = svd.pseudoinverse();
    let y = Col::from_iter(
        data.iter()
            .map(|LabeledPoint { label, .. }| f64::from(*label)),
    );
    (
        ((Mat::<f64, _, _>::identity(data.len(), data.len()) - mat * pinv.as_ref()) * y.as_ref())
            .squared_norm_l2()
            / (data.len() as f64),
        pinv * y,
    )
}

fn q12() -> f64 {
    use rayon::iter::{IntoParallelIterator as _, ParallelIterator as _};
    const NUM_EXPS: usize = 1000;
    let data = augment_data(get_data::<DIM, _>("./hw3_train.txt"), 1.0);
    let mut mat = Mat::<f64, _, _>::zeros(data.len(), DIM + 1);
    mat.row_iter_mut()
        .zip(&data)
        .for_each(|(row, LabeledPoint { coordinates, .. })| {
            row.iter_mut()
                .zip(coordinates)
                .for_each(|(coordinate, c)| *coordinate = *c);
        });
    let mat = mat.as_ref();
    let y = Col::from_iter(
        data.iter()
            .map(|LabeledPoint { label, .. }| f64::from(*label)),
    );

    (0..NUM_EXPS)
        .into_par_iter()
        .map_init(rand::rng, |mut rng, _experiment| {
            use rand::prelude::*;
            const ETA: f64 = 0.001;
            let w = (0..800).fold(Col::zeros(DIM + 1), |w, _iterations| {
                let LabeledPoint { coordinates, label } = data.choose(&mut rng).unwrap();
                let x = ColRef::from_slice(coordinates);
                Scale(ETA * 2.0 * (f64::from(*label) - w.transpose() * x)) * x + w
            });
            (mat * MatRef::from_repeated_col(w.as_ref(), 1)
                - MatRef::from_repeated_col(y.as_ref(), 1))
            .squared_norm_l2()
                / (data.len() as f64)
        })
        .sum::<f64>()
        / (NUM_EXPS as f64)
}

fn q13() -> f64 {
    sgd_logistic(Col::zeros(DIM + 1).as_ref())
}

fn q14(init_w: ColRef<f64>) -> f64 {
    sgd_logistic(init_w)
}

fn sgd_logistic(init_w: ColRef<f64>) -> f64 {
    use rayon::iter::{IntoParallelIterator as _, ParallelIterator as _};
    const NUM_EXPS: usize = 1000;
    let data = augment_data(get_data::<DIM, _>("./hw3_train.txt"), 1.0);

    (0..NUM_EXPS)
        .into_par_iter()
        .map_init(rand::rng, |mut rng, _experiment| {
            use rand::prelude::*;
            const ETA: f64 = 0.001;
            let w = (0..800).fold(init_w.to_owned(), |w, _iterations| {
                let LabeledPoint { coordinates, label } = data.choose(&mut rng).unwrap();
                let x = ColRef::from_slice(coordinates);
                Scale(ETA * f64::from(*label) * pw_sgd(*label, w.as_ref(), x)) * x + w
            });

            data.iter()
                .fold(0.0, |err, LabeledPoint { coordinates, label }| {
                    err + f64::ln(
                        1.0 + f64::exp(
                            -f64::from(*label) * w.transpose() * ColRef::from_slice(coordinates),
                        ),
                    )
                })
                / (data.len() as f64)
        })
        .sum::<f64>()
        / (NUM_EXPS as f64)
}

fn main() {
    let (q11, q11w) = q11();
    println!(r#"q11: E_"in" = {}"#, q11);
    println!(r#"q12: E_"in" = {}"#, q12());
    println!(r#"q13: E_"in" = {}"#, q13());
    println!(r#"q14: E_"in" = {}"#, q14(q11w.as_ref()));
}
