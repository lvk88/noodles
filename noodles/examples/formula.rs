use noodles::{contouring::contour_formula, grid::SizedGrid};

fn main() {
    let heart = "(x^2 + y^2 - 1.)^3 - x^2 * y^3";

    let grid = SizedGrid::create_sized_grid([-2.5, -2.5], [5.0, 5.0], [1000, 1000]);
    let res = contour_formula(heart, grid).expect("Failed to run segmenting!");
    res.write_obj("formula.obj");
}
