fn main() {
    //fn circle(x: f32, y: f32) -> f32 {
    //    let cx = 2.5;
    //    let cy = 2.5;

    //    let r = 15. / 8.;

    //    (x - cx) * (x - cx) + (y - cy) * (y - cy) - r * r
    //}

    #[allow(non_snake_case)]
    fn ring(x: f32, y: f32) -> f32 {
        let cx = 2.5;
        let cy = 2.5;

        let R = 13. / 8.;
        let r = 5. / 8.;

        let c = (x - cx) * (x - cx) + (y - cy) * (y - cy) - r * r;
        let C = (x - cx) * (x - cx) + (y - cy) * (y - cy) - R * R;

        f32::max(C, -c)
    }

    //let sized_grid = noodles::grid::SizedGrid::new(
    //    [0.0, 0.0],
    //    [5. / 4., 5. / 4.],
    //    noodles::grid::Grid::new_from_size(4, 4),
    //);

    let sized_grid =
        noodles::grid::SizedGrid::create_sized_grid([0.0, 0.0], [5.0, 5.0], [1000, 1000]);

    let result = noodles::contouring::contour(&ring, sized_grid);

    result.write_obj("test.obj");
}
