use std::{fs::File, io::read_to_string, path::Path};

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

fn q11() -> f64 {
    const DIM: usize = 10;
    let data = augment_data(get_data::<DIM, _>("./hw3_train.txt"), 1.0);
    let mut mat = faer::mat::Mat::<f64, _, _>::zeros(data.len(), DIM + 1);
    mat.row_iter_mut()
        .zip(&data)
        .for_each(|(row, LabeledPoint { coordinates, .. })| {
            row.iter_mut()
                .zip(coordinates)
                .for_each(|(coordinate, c)| *coordinate = *c);
        });
    let svd = faer::linalg::solvers::Svd::new(mat.as_ref()).unwrap();
    let pinv = svd.pseudoinverse();
    let y = faer::col::Col::from_iter(data.iter().map(|LabeledPoint { label, .. }| match label {
        Label::Positive => 1.0,
        Label::Negative => -1.0,
    }));
    return ((faer::mat::Mat::<f64, _, _>::identity(data.len(), data.len()) - mat * pinv) * y)
        .squared_norm_l2()
        / (data.len() as f64);
}

fn main() {
    println!(r#"q11: E_"in" = {}"#, q11());
}
