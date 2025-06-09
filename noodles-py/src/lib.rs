use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass]
struct SizedGrid {
    #[pyo3(get)]
    origin: [f32; 2],

    #[pyo3(get)]
    sizes: [f32; 2],

    #[pyo3(get)]
    n_cells: [u32; 2],
}

#[pymethods]
impl SizedGrid {
    #[new]
    fn new(origin: [f32; 2], sizes: [f32; 2], n_cells: [u32; 2]) -> Self {
        SizedGrid {
            origin,
            sizes,
            n_cells,
        }
    }
}

impl Into<noodles::grid::SizedGrid> for &SizedGrid {
    fn into(self) -> noodles::grid::SizedGrid {
        noodles::grid::SizedGrid::create_sized_grid(self.origin, self.sizes, self.n_cells)
    }
}

#[pyclass]
struct Contour {
    #[pyo3(get)]
    pub points: Vec<[f32; 2]>,
    #[pyo3(get)]
    pub edges: Vec<(usize, usize)>,
}

impl From<noodles::tessellation::Contour> for Contour {
    fn from(value: noodles::tessellation::Contour) -> Self {
        let points = value.points.iter().map(|p| [p.x, p.y]).collect();
        let edges = value.edges;
        Self { points, edges }
    }
}

#[pyfunction]
fn hello() {
    println!("Hello world!");
}

#[pyfunction]
fn contour(formula: &str, grid: &SizedGrid) -> PyResult<Contour> {
    let sized_grid: noodles::grid::SizedGrid = grid.into();
    let contour = noodles::contouring::contour_formula(formula, sized_grid);

    if let Err(_) = contour {
        return Err(PyValueError::new_err("Failed to parse formula"));
    }

    let contour: Contour = contour.unwrap().into();
    Ok(contour)
}

#[pymodule]
fn noodles_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(contour, m)?)?;
    m.add_class::<SizedGrid>()?;
    m.add_class::<Contour>()?;
    Ok(())
}
