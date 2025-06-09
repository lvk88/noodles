use serde::Serialize;
use serde_json::Result;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct SizedGrid {
    origin: [f32; 2],
    sizes: [f32; 2],
    n_cells: [u32; 2],
}

#[wasm_bindgen]
impl SizedGrid {
    #[wasm_bindgen(constructor)]
    pub fn new(
        origin_x: f32,
        origin_y: f32,
        size_x: f32,
        size_y: f32,
        n_cells_x: u32,
        n_cells_y: u32,
    ) -> SizedGrid {
        SizedGrid {
            origin: [origin_x, origin_y],
            sizes: [size_x, size_y],
            n_cells: [n_cells_x, n_cells_y],
        }
    }
}

impl Into<noodles::grid::SizedGrid> for &SizedGrid {
    fn into(self) -> noodles::grid::SizedGrid {
        noodles::grid::SizedGrid::create_sized_grid(self.origin, self.sizes, self.n_cells)
    }
}

#[wasm_bindgen]
#[derive(Serialize)]
pub struct Polygon {
    points: Vec<(f32, f32)>,
    edges: Vec<(usize, usize)>,
}

#[wasm_bindgen]
impl Polygon {
    pub fn n_points(&self) -> usize {
        self.points.len()
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn serialize(&self) -> String {
        let res = serde_json::to_string(&self).unwrap();
        res
    }
}

impl From<noodles::tessellation::Contour> for Polygon {
    fn from(value: noodles::tessellation::Contour) -> Self {
        let points = value.points.iter().map(|p| (p.x, p.y)).collect();
        let edges = value.edges;
        Self { points, edges }
    }
}

#[wasm_bindgen]
pub fn contour(expr: &str, grid: &SizedGrid) -> Polygon {
    let grid: noodles::grid::SizedGrid = grid.into();
    let res: Polygon = noodles::contouring::contour_formula(expr, grid)
        .expect("Could not execute")
        .into();
    res
}
