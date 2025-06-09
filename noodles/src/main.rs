fn main() {
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

    let sized_grid =
        noodles::grid::SizedGrid::create_sized_grid([0.0, 0.0], [5.0, 5.0], [100, 100]);
    let result = noodles::contouring::contour(&ring, sized_grid);
    result.write_obj("test.obj");
}
