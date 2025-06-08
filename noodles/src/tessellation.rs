use std::fs::File;

#[derive(Debug)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
}

#[derive(Default, Debug)]
pub struct Contour {
    pub points: Vec<Point2D>,
    pub edges: Vec<(usize, usize)>,
}

impl Contour {
    pub fn write_obj(&self, file_name: &str) {
        use std::io::Write;
        let mut file = File::create(file_name).unwrap();

        for p in &self.points {
            writeln!(&mut file, "v {} {} 0.0", p.x, p.y).expect("Could not write string to file");
        }

        for e in &self.edges {
            writeln!(&mut file, "l {} {}", e.0 + 1, e.1 + 1)
                .expect("Could not write string to file");
        }
    }
}
