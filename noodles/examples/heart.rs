fn main() {
    let grid =
        noodles::grid::SizedGrid::create_sized_grid([-5.0, -5.0], [10.0, 10.0], [1001, 1001]);

    fn heart(x: f32, y: f32) -> f32 {
        let res = x * x + y * y - 1.0;
        let res = res * res * res - x * x * y * y * y;
        res
    }

    let res = noodles::contouring::contour(&heart, grid);

    res.write_obj("heart.obj");
}
