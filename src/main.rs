use faer::{
    Scale,
    col::{Col, ColRef},
    mat::{Mat, MatRef},
    row::RowRef,
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
impl Label {
    fn same_sign(&self, f: f64) -> bool {
        match (self, f.is_sign_positive()) {
            (Self::Positive, true) | (Self::Negative, false) => true,
            _ => false,
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
    let train = augment_data(get_data::<DIM, _>("./hw3_train.txt"), 1.0);
    let mut mat = Mat::<f64, _, _>::zeros(train.len(), DIM + 1);
    mat.row_iter_mut()
        .zip(&train)
        .for_each(|(row, LabeledPoint { coordinates, .. })| {
            row.iter_mut()
                .zip(coordinates)
                .for_each(|(coordinate, c)| *coordinate = *c);
        });
    let svd = faer::linalg::solvers::Svd::new(mat.as_ref()).unwrap();
    let pinv = svd.pseudoinverse();
    let y = Col::from_iter(
        train
            .iter()
            .map(|LabeledPoint { label, .. }| f64::from(*label)),
    );
    (
        ((Mat::<f64, _, _>::identity(train.len(), train.len()) - mat * pinv.as_ref()) * y.as_ref())
            .squared_norm_l2()
            / (train.len() as f64),
        pinv * y,
    )
}

fn q12() -> f64 {
    use rayon::iter::{IntoParallelIterator as _, ParallelIterator as _};
    const NUM_EXPS: usize = 1000;
    let train = augment_data(get_data::<DIM, _>("./hw3_train.txt"), 1.0);
    let mut mat = Mat::<f64, _, _>::zeros(train.len(), DIM + 1);
    mat.row_iter_mut()
        .zip(&train)
        .for_each(|(row, LabeledPoint { coordinates, .. })| {
            row.iter_mut()
                .zip(coordinates)
                .for_each(|(coordinate, c)| *coordinate = *c);
        });
    let mat = mat.as_ref();
    let y = Col::from_iter(
        train
            .iter()
            .map(|LabeledPoint { label, .. }| f64::from(*label)),
    );

    (0..NUM_EXPS)
        .into_par_iter()
        .map_init(rand::rng, |mut rng, _experiment| {
            use rand::prelude::*;
            const ETA: f64 = 0.001;
            let w = (0..800).fold(Col::zeros(DIM + 1), |w, _iterations| {
                let LabeledPoint { coordinates, label } = train.choose(&mut rng).unwrap();
                let x = ColRef::from_slice(coordinates);
                Scale(ETA * 2.0 * (f64::from(*label) - w.transpose() * x)) * x + w
            });
            (mat * MatRef::from_repeated_col(w.as_ref(), 1)
                - MatRef::from_repeated_col(y.as_ref(), 1))
            .squared_norm_l2()
                / (train.len() as f64)
        })
        .sum::<f64>()
        / (NUM_EXPS as f64)
}

fn q13() -> f64 {
    sgd_logistic(Col::zeros(DIM + 1).as_ref()).0
}

fn q14_q15(init_w: ColRef<f64>) -> (f64, f64) {
    sgd_logistic(init_w)
}

fn binary_err(w: ColRef<f64>, data: &[LabeledPoint]) -> f64 {
    data.iter()
        .filter(|LabeledPoint { coordinates, label }| {
            !label.same_sign(RowRef::from_slice(coordinates) * w)
        })
        .count() as f64
        / (data.len() as f64)
}

fn sgd_logistic(init_w: ColRef<f64>) -> (f64, f64) {
    use rayon::iter::{IntoParallelIterator as _, ParallelIterator as _};
    const NUM_EXPS: usize = 1000;
    let train = augment_data(get_data::<DIM, _>("./hw3_train.txt"), 1.0);

    // FIXME doing Q15 0/1 error here is such a messy code placement/pipeline...
    let test = augment_data(get_data::<DIM, _>("./hw3_test.txt"), 1.0);

    #[derive(Clone, Copy, Debug)]
    struct HW3 {
        q15: f64,
        e_in: f64,
    }
    impl std::iter::Sum for HW3 {
        /// TODO this abuses the fields: it's at the same time one experiment result and sum
        /// maybe sum for heterogeneous types...?
        fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.into_iter().fold(
                HW3 {
                    q15: 0.0,
                    e_in: 0.0,
                },
                |HW3 {
                     q15: q15_sum,
                     e_in: e_in_sum,
                 },
                 HW3 { q15, e_in }| {
                    HW3 {
                        q15: q15_sum + q15,
                        e_in: e_in_sum + e_in,
                    }
                },
            )
        }
    }

    let ret = (0..NUM_EXPS)
        .into_par_iter()
        .map_init(rand::rng, |mut rng, _experiment| {
            use rand::prelude::*;
            const ETA: f64 = 0.001;
            let w = (0..800).fold(init_w.to_owned(), |w, _iterations| {
                let LabeledPoint { coordinates, label } = train.choose(&mut rng).unwrap();
                let x = ColRef::from_slice(coordinates);
                Scale(ETA * f64::from(*label) * pw_sgd(*label, w.as_ref(), x)) * x + w
            });

            HW3 {
                q15: f64::abs(binary_err(w.as_ref(), &train) - binary_err(w.as_ref(), &test)),
                e_in: train
                    .iter()
                    .fold(0.0, |err, LabeledPoint { coordinates, label }| {
                        err + f64::ln(
                            1.0 + f64::exp(
                                -f64::from(*label)
                                    * w.transpose()
                                    * ColRef::from_slice(coordinates),
                            ),
                        )
                    })
                    / (train.len() as f64),
            }
        })
        .sum::<HW3>();
    (ret.e_in / (NUM_EXPS as f64), ret.q15 / (NUM_EXPS as f64))
}

fn q16(w: ColRef<f64>) -> f64 {
    let train = augment_data(get_data::<DIM, _>("./hw3_train.txt"), 1.0);
    let test = augment_data(get_data::<DIM, _>("./hw3_test.txt"), 1.0);
    f64::abs(binary_err(w, &train) - binary_err(w, &test))
}

fn main() {
    let (q11, q11w) = q11();
    println!(r#"q11: E_"in" = {}"#, q11);
    println!(r#"q12: E_"in" = {}"#, q12());
    println!(r#"q13: E_"in" = {}"#, q13());
    let (q14, q15) = q14_q15(q11w.as_ref());
    println!(r#"q14: E_"in" = {}"#, q14);
    println!(r#"q15: generalization error = {}"#, q15);
    println!(r#"q16: generalization error = {}"#, q16(q11w.as_ref()));
}
