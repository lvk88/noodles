use crate::contouring::{case, contour};

#[macro_use]
extern crate assert_float_eq;

mod tessellation {
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
                writeln!(&mut file, "v {} {} 0.0", p.x, p.y);
            }

            for e in &self.edges {
                writeln!(&mut file, "l {} {}", e.0 + 1, e.1 + 1);
            }
        }
    }
}

mod grid {
    pub struct Grid {
        N: u32,
        M: u32,
    }

    #[derive(PartialEq)]
    pub enum EdgeDirection {
        Horizontal,
        Vertical,
    }

    impl Grid {
        pub fn new() -> Grid {
            Grid { N: 0, M: 0 }
        }

        pub fn N(&self) -> u32 {
            self.N
        }

        pub fn M(&self) -> u32 {
            self.M
        }

        pub fn new_from_size(N: u32, M: u32) -> Grid {
            Grid { N, M }
        }

        pub fn ravel(&self, i: u32, j: u32) -> u32 {
            j * self.N + i
        }

        pub fn unravel(&self, index: u32) -> (u32, u32) {
            let j = index / self.N;
            let i = index % self.N;
            (i, j)
        }

        pub fn cell_nodes(&self, i: u32, j: u32) -> [u32; 4] {
            [
                j * (self.N + 1) + i,
                j * (self.N + 1) + i + 1,
                (j + 1) * (self.N + 1) + i,
                (j + 1) * (self.N + 1) + i + 1,
            ]
        }

        pub fn cell_node_indices(&self, i: u32, j: u32) -> [(u32, u32); 4] {
            [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
        }

        pub fn cell_edges(&self, i: u32, j: u32) -> [u32; 4] {
            [
                j * (2 * self.N + 1) + i,
                j * (2 * self.N + 1) + self.N + i + 1,
                (j + 1) * (2 * self.N + 1) + i,
                j * (2 * self.N + 1) + self.N + i,
            ]
        }

        // Returns the origin and direction of the edge identified by ID `e`
        // Direction may be horizontal or vertical
        pub fn edge(&self, e: u32) -> ((u32, u32), EdgeDirection) {
            let j = e / (2 * self.N + 1);
            let i = e % (2 * self.N + 1);

            if i > (self.N - 1) {
                // Vertical edge
                if i == 2 * self.N {
                    ((self.N, j), EdgeDirection::Vertical)
                } else {
                    ((i % self.N, j), EdgeDirection::Vertical)
                }
            } else {
                // Horizontal edge
                ((i, j), EdgeDirection::Horizontal)
            }
        }

        // Returns the two cell id-s that are shared by an edge and also the id of the edge in each cell
        pub fn edge_to_cell(&self, e: u32) -> [(u32, u32); 2] {
            let j = e / (2 * self.N + 1);
            let i = e % (2 * self.N + 1);
            let [cell_1, cell_2] = if i > self.N - 1 {
                // This is a vertical edge
                let cell_id = self.ravel(i % self.N - 1, j);
                [(cell_id, 1), (cell_id + 1, 3)]
            } else {
                // This is a horizontal edge
                let cell_id = self.ravel(i, j - 1);
                [(cell_id, 2), (cell_id + self.N, 0)]
            };
            [cell_1, cell_2]
        }
    }

    pub struct SizedGrid {
        grid: Grid,
        origin: [f32; 2],
        delta: [f32; 2],
    }

    impl SizedGrid {
        pub fn new(origin: [f32; 2], delta: [f32; 2], grid: Grid) -> SizedGrid {
            SizedGrid {
                grid,
                origin,
                delta,
            }
        }

        pub fn point(&self, i: u32, j: u32) -> [f32; 2] {
            let x = self.origin[0] + i as f32 * self.delta[0];
            let y = self.origin[1] + j as f32 * self.delta[1];
            [x, y]
        }

        pub fn grid(&self) -> &Grid {
            &self.grid
        }

        pub fn delta_i(&self, i: usize) -> f32 {
            self.delta[i]
        }
    }
}

mod contouring {

    use std::{cell, collections::HashMap, hash::Hash, iter::Enumerate};

    use crate::{
        grid::{EdgeDirection, SizedGrid},
        tessellation::{self, Point2D},
    };
    const LOCAL_EDGE_TO_INTERNAL_POINT: [[u32; 4]; 16] = [
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 2, 2, 1],
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 2, 2],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 0],
    ];

    pub type ImplicitFunction = fn(x: f32, y: f32) -> f32;

    pub fn case<F>(sized_grid: &SizedGrid, i: u32, j: u32, func: &F) -> u32
    where
        F: Fn(f32, f32) -> f32,
    {
        let nodes = sized_grid.grid().cell_node_indices(i, j);

        let points = nodes.map(|(i, j)| sized_grid.point(i, j));

        let _vals = points.map(|point| func(point[0], point[1]));

        let states = points.map(|point| func(point[0], point[1]) <= f32::EPSILON);

        let case = states.iter().enumerate().fold(0, |acc, (index, &state)| {
            if state {
                acc + u32::pow(2, index as u32)
            } else {
                acc
            }
        });

        case
    }

    // Given an edge id of an intersected edge, returns the cell id-s and the internal point id-s for the two internal points which define the dual edge
    pub fn segment<F>(sized_grid: &SizedGrid, e: u32, func: &F) -> ((u32, u32), (u32, u32))
    where
        F: Fn(f32, f32) -> f32,
    {
        let cells = sized_grid.grid().edge_to_cell(e);

        let cell_0 = sized_grid.grid().unravel(cells[0].0);
        let cell_1 = sized_grid.grid().unravel(cells[1].0);

        let case_0 = case(sized_grid, cell_0.0, cell_0.1, func);
        let case_1 = case(sized_grid, cell_1.0, cell_1.1, func);

        let internal_point_0 = LOCAL_EDGE_TO_INTERNAL_POINT[case_0 as usize][cells[0].1 as usize];
        let internal_point_1 = LOCAL_EDGE_TO_INTERNAL_POINT[case_1 as usize][cells[1].1 as usize];

        (
            (cells[0].0, internal_point_0),
            (cells[1].0, internal_point_1),
        )
    }

    pub fn intersect<F>(sized_grid: &SizedGrid, edge: u32, func: &F) -> (f32, f32)
    where
        F: Fn(f32, f32) -> f32,
    {
        let edge = sized_grid.grid().edge(edge);

        let p0 = sized_grid.point(edge.0.0, edge.0.1);
        let p1 = match edge.1 {
            EdgeDirection::Horizontal => sized_grid.point(edge.0.0 + 1, edge.0.1),
            EdgeDirection::Vertical => sized_grid.point(edge.0.0, edge.0.1 + 1),
        };

        let v0 = func(p0[0], p0[1]);
        let v1 = func(p1[0], p1[1]);

        let dy = v1 - v0;

        let dt = match edge.1 {
            EdgeDirection::Horizontal => sized_grid.delta_i(0),
            EdgeDirection::Vertical => sized_grid.delta_i(1),
        };

        let dr = -v0 * dt / (v1 - v0);

        match edge.1 {
            EdgeDirection::Horizontal => (p0[0] + dr, p0[1]),
            EdgeDirection::Vertical => (p0[0], p0[1] + dr),
        }
    }

    pub fn topo_segment<F>(f: &F, grid: &SizedGrid) -> (Vec<((u32, u32), (u32, u32))>, Vec<u32>)
    where
        F: Fn(f32, f32) -> f32,
    {
        let N = grid.grid().N();
        let M = grid.grid().M();

        // Contains edges, which are defined by two internal points in two neighboring cells
        let mut segments: Vec<((u32, u32), (u32, u32))> = Default::default();
        let mut intersected_edges: Vec<u32> = Default::default();

        // Iterate over all edges, and if there is an intersected edge store the connectivity in a map
        for edge_index in 0..(M * (2 * N + 1) + N) {
            let edge = grid.grid().edge(edge_index);
            let p0 = grid.point(edge.0.0, edge.0.1);
            let p1 = match edge.1 {
                EdgeDirection::Horizontal => grid.point(edge.0.0 + 1, edge.0.1),
                EdgeDirection::Vertical => grid.point(edge.0.0, edge.0.1 + 1),
            };

            let v0 = f(p0[0], p0[1]) < f32::EPSILON;
            let v1 = f(p1[0], p1[1]) < f32::EPSILON;

            // Bad idea to compute product. We need to analyze the same way as we do when identifying the cell cases, i.e. compare the boolean values
            //if v0 * v1 > f32::EPSILON {
            //    continue;
            //}

            if v0 == v1 {
                continue;
            }

            // Edge is intersected, we can put a segment in the edges vector
            let segment = segment(&grid, edge_index, &f);
            segments.push(segment);
            intersected_edges.push(edge_index);
        }
        (segments, intersected_edges)
    }

    pub fn internal_points<F>(f: &F, grid: &SizedGrid) -> HashMap<(u32, u32), (f32, f32)>
    where
        F: Fn(f32, f32) -> f32,
    {
        let mut res: HashMap<(u32, u32), (f32, f32)> = Default::default();

        let N = grid.grid().N();
        let M = grid.grid().M();

        for cell_j in 0..M {
            for cell_i in 0..N {
                let cell_index = cell_j * N + cell_i;
                let case = case(&grid, cell_i, cell_j, &f);

                let cell_edges = grid.grid().cell_edges(cell_i, cell_j);

                let edge_to_internal = LOCAL_EDGE_TO_INTERNAL_POINT[case as usize];
                let n_internal_points = edge_to_internal
                    .iter()
                    .filter(|&edge_id| *edge_id != 0)
                    .count();
                let n_internal_points = edge_to_internal.iter().max().unwrap();
                let mut internal_points: Vec<(f32, f32)> =
                    vec![(0., 0.); *n_internal_points as usize];
                for (local_edge, global_edge) in cell_edges.iter().enumerate() {
                    let internal_point_id = edge_to_internal[local_edge];
                    if internal_point_id == 0 {
                        continue;
                    }
                    let intersection = intersect(&grid, *global_edge, f);
                    internal_points[(internal_point_id - 1) as usize].0 += intersection.0;
                    internal_points[(internal_point_id - 1) as usize].1 += intersection.1;
                }

                for (i, internal_point) in internal_points.iter_mut().enumerate() {
                    internal_point.0 /= 2.;
                    internal_point.1 /= 2.;
                    res.insert((cell_index, (i + 1) as u32), *internal_point);
                }
            }
        }

        res
    }

    pub fn contour<F>(f: F, grid: SizedGrid) -> super::tessellation::Contour
    where
        F: Fn(f32, f32) -> f32,
    {
        // Create topological connectivities
        let (segments, intersected_edges) = topo_segment(&f, &grid);
        println!("{:?}", &segments);

        // Create internal points
        let internal_points = internal_points(&f, &grid);

        let mut res = tessellation::Contour::default();

        let mut internal_point_to_vertex: HashMap<(u32, u32), u32> = Default::default();

        for internal_point in internal_points {
            let (ids, coords) = internal_point;
            res.points.push(Point2D {
                x: coords.0,
                y: coords.1,
            });
            internal_point_to_vertex.insert(ids, (res.points.len() - 1) as u32);
        }

        for segment in segments {
            let start_vertex = internal_point_to_vertex.get(&segment.0).unwrap();
            let end_vertex = internal_point_to_vertex.get(&segment.1).unwrap();
            res.edges
                .push((*start_vertex as usize, *end_vertex as usize));
        }

        res
    }
}

fn main() {
    fn circle(x: f32, y: f32) -> f32 {
        let cx = 2.5;
        let cy = 2.5;

        let r = 15. / 8.;

        (x - cx) * (x - cx) + (y - cy) * (y - cy) - r * r
    }

    fn ring(x: f32, y: f32) -> f32 {
        let cx = 2.5;
        let cy = 2.5;

        let R = 13. / 8.;
        let r = 5. / 8.;

        let c = (x - cx) * (x - cx) + (y - cy) * (y - cy) - r * r;
        let C = (x - cx) * (x - cx) + (y - cy) * (y - cy) - R * R;

        f32::max(C, -c)
    }

    let sized_grid = grid::SizedGrid::new(
        [0.0, 0.0],
        [5. / 4., 5. / 4.],
        grid::Grid::new_from_size(4, 4),
    );

    let cell_case = case(&sized_grid, 1, 1, &ring);
    println!("{}", cell_case);

    let result = contour(&ring, sized_grid);

    result.write_obj("test.obj");
}

#[cfg(test)]
mod tests {
    use crate::{
        contouring::{ImplicitFunction, case, internal_points, intersect, segment, topo_segment},
        grid::{Grid, SizedGrid},
    };

    use super::*;

    #[test]
    fn test_default() {
        let default = tessellation::Contour::default();
        assert_eq!(default.points.len(), 0);
        assert_eq!(default.edges.len(), 0);
    }

    #[test]
    fn test_grid_ravel() {
        let g = grid::Grid::new_from_size(2, 3);

        assert_eq!(g.ravel(1, 1), 3);
    }

    #[test]
    fn test_cell_nodes() {
        let g = grid::Grid::new_from_size(2, 3);

        let cell_nodes = g.cell_nodes(1, 1);
        assert_eq!(cell_nodes[0], 4);
        assert_eq!(cell_nodes[1], 5);
        assert_eq!(cell_nodes[2], 7);
        assert_eq!(cell_nodes[3], 8);
    }

    #[test]
    fn test_cell_node_indices() {
        let g = grid::Grid::new_from_size(2, 3);

        let cell_node_indices = g.cell_node_indices(1, 1);

        assert_eq!(cell_node_indices[0].0, 1);
        assert_eq!(cell_node_indices[0].1, 1);

        assert_eq!(cell_node_indices[1].0, 2);
        assert_eq!(cell_node_indices[1].1, 1);

        assert_eq!(cell_node_indices[2].0, 2);
        assert_eq!(cell_node_indices[2].1, 2);

        assert_eq!(cell_node_indices[3].0, 1);
        assert_eq!(cell_node_indices[3].1, 2);
    }

    #[test]
    fn test_cell_edges() {
        let g = grid::Grid::new_from_size(2, 3);

        let edges = g.cell_edges(1, 1);

        assert_eq!(edges[0], 6);
        assert_eq!(edges[1], 9);
        assert_eq!(edges[2], 11);
        assert_eq!(edges[3], 8);
    }

    #[test]
    fn test_edge() {
        let g = grid::Grid::new_from_size(6, 4);

        // Horizontal edge
        let e = g.edge(17);
        assert_eq!(e.0.0, 4);
        assert_eq!(e.0.1, 1);
        assert!(e.1 == grid::EdgeDirection::Horizontal);

        // Horizontal edge
        let e = g.edge(13);
        assert_eq!(e.0.0, 0);
        assert_eq!(e.0.1, 1);
        assert!(e.1 == grid::EdgeDirection::Horizontal);

        // Vertical edge
        let e = g.edge(25);
        assert_eq!(e.0.0, 6);
        assert_eq!(e.0.1, 1);
        assert!(e.1 == grid::EdgeDirection::Vertical);

        // Vertical edge
        let e = g.edge(38);
        assert_eq!(e.0.0, 6);
        assert_eq!(e.0.1, 2);
        assert!(e.1 == grid::EdgeDirection::Vertical);
    }

    #[test]
    fn test_edge_to_cell() {
        let g = grid::Grid::new_from_size(6, 4);

        // Vertical edge
        let neighbors = g.edge_to_cell(22);
        assert_eq!(neighbors[0].0, 8);
        assert_eq!(neighbors[0].1, 1);
        assert_eq!(neighbors[1].0, 9);
        assert_eq!(neighbors[1].1, 3);

        // Horizontal edge
        let neighbors = g.edge_to_cell(15);
        assert_eq!(neighbors[0].0, 2);
        assert_eq!(neighbors[0].1, 2);
        assert_eq!(neighbors[1].0, 8);
        assert_eq!(neighbors[1].1, 0);
    }

    #[test]
    fn test_grid_point() {
        let sized_grid = SizedGrid::new([1.0, 2.0], [3.0, 4.0], Grid::new_from_size(2, 3));

        let point = sized_grid.point(2, 1);
        assert_f32_near!(point[0], 7.0);
        assert_f32_near!(point[1], 6.0);
    }

    #[test]
    fn test_get_case() {
        let sized_grid = SizedGrid::new([0.0, 0.0], [5. / 4., 5. / 4.], Grid::new_from_size(4, 4));

        fn circle(x: f32, y: f32) -> f32 {
            let cx = 2.5;
            let cy = 2.5;

            let r = 15. / 8.;

            (x - cx) * (x - cx) + (y - cy) * (y - cy) - r * r
        }

        // j = 0
        let mut cell_case = case(&sized_grid, 0, 0, &circle);
        assert_eq!(cell_case, 4);

        cell_case = case(&sized_grid, 1, 0, &circle);
        assert_eq!(cell_case, 12);

        cell_case = case(&sized_grid, 2, 0, &circle);
        assert_eq!(cell_case, 12);

        cell_case = case(&sized_grid, 3, 0, &circle);
        assert_eq!(cell_case, 8);

        // j = 1
        cell_case = case(&sized_grid, 0, 1, &circle);
        assert_eq!(cell_case, 6);

        cell_case = case(&sized_grid, 1, 1, &circle);
        assert_eq!(cell_case, 15);

        cell_case = case(&sized_grid, 2, 1, &circle);
        assert_eq!(cell_case, 15);

        cell_case = case(&sized_grid, 3, 1, &circle);
        assert_eq!(cell_case, 9);

        // j = 2
        cell_case = case(&sized_grid, 0, 2, &circle);
        assert_eq!(cell_case, 6);

        cell_case = case(&sized_grid, 1, 2, &circle);
        assert_eq!(cell_case, 15);

        cell_case = case(&sized_grid, 2, 2, &circle);
        assert_eq!(cell_case, 15);

        cell_case = case(&sized_grid, 3, 2, &circle);
        assert_eq!(cell_case, 9);

        // j = 3
        cell_case = case(&sized_grid, 0, 3, &circle);
        assert_eq!(cell_case, 2);

        cell_case = case(&sized_grid, 1, 3, &circle);
        assert_eq!(cell_case, 3);

        cell_case = case(&sized_grid, 2, 3, &circle);
        assert_eq!(cell_case, 3);

        cell_case = case(&sized_grid, 3, 3, &circle);
        assert_eq!(cell_case, 1);
    }

    #[test]
    fn test_segment() {
        let sized_grid = SizedGrid::new([0.0, 0.0], [5. / 4., 5. / 4.], Grid::new_from_size(4, 4));

        fn circle(x: f32, y: f32) -> f32 {
            let cx = 2.5;
            let cy = 2.5;

            let r = 15. / 8.;

            (x - cx) * (x - cx) + (y - cy) * (y - cy) - r * r
        }

        let edge = 18;
        let segment = segment(&sized_grid, edge, &circle);

        assert_eq!(segment.0.0, 4);
        assert_eq!(segment.0.1, 1);
        assert_eq!(segment.1.0, 8);
        assert_eq!(segment.1.1, 1);
    }

    #[test]
    fn test_topo_segment() {
        let sized_grid = SizedGrid::new([0.0, 0.0], [5. / 4., 5. / 4.], Grid::new_from_size(4, 4));

        fn circle(x: f32, y: f32) -> f32 {
            let cx = 2.5;
            let cy = 2.5;

            let r = 15. / 8.;

            (x - cx) * (x - cx) + (y - cy) * (y - cy) - r * r
        }

        let (segments, intersected_edges) = topo_segment(&circle, &sized_grid);

        assert_eq!(segments.len(), 12);

        // TODO: a better comparison
        assert_eq!(segments[0].0.0, 0);
        assert_eq!(segments[0].0.1, 1);
        assert_eq!(segments[0].1.0, 1);
        assert_eq!(segments[0].1.1, 1);

        assert_eq!(segments[1].0.0, 1);
        assert_eq!(segments[1].0.1, 1);
        assert_eq!(segments[1].1.0, 2);
        assert_eq!(segments[1].1.1, 1);

        assert_eq!(segments[2].0.0, 2);
        assert_eq!(segments[2].0.1, 1);
        assert_eq!(segments[2].1.0, 3);
        assert_eq!(segments[2].1.1, 1);

        assert_eq!(segments[3].0.0, 0);
        assert_eq!(segments[3].0.1, 1);
        assert_eq!(segments[3].1.0, 4);
        assert_eq!(segments[3].1.1, 1);

        assert_eq!(segments[4].0.0, 3);
        assert_eq!(segments[4].0.1, 1);
        assert_eq!(segments[4].1.0, 7);
        assert_eq!(segments[4].1.1, 1);

        assert_eq!(segments[5].0.0, 4);
        assert_eq!(segments[5].0.1, 1);
        assert_eq!(segments[5].1.0, 8);
        assert_eq!(segments[5].1.1, 1);

        assert_eq!(segments[6].0.0, 7);
        assert_eq!(segments[6].0.1, 1);
        assert_eq!(segments[6].1.0, 11);
        assert_eq!(segments[6].1.1, 1);

        assert_eq!(segments[7].0.0, 8);
        assert_eq!(segments[7].0.1, 1);
        assert_eq!(segments[7].1.0, 12);
        assert_eq!(segments[7].1.1, 1);

        assert_eq!(segments[8].0.0, 11);
        assert_eq!(segments[8].0.1, 1);
        assert_eq!(segments[8].1.0, 15);
        assert_eq!(segments[8].1.1, 1);

        assert_eq!(segments[9].0.0, 12);
        assert_eq!(segments[9].0.1, 1);
        assert_eq!(segments[9].1.0, 13);
        assert_eq!(segments[9].1.1, 1);

        assert_eq!(segments[10].0.0, 13);
        assert_eq!(segments[10].0.1, 1);
        assert_eq!(segments[10].1.0, 14);
        assert_eq!(segments[10].1.1, 1);

        assert_eq!(segments[11].0.0, 14);
        assert_eq!(segments[11].0.1, 1);
        assert_eq!(segments[11].1.0, 15);
        assert_eq!(segments[11].1.1, 1);
    }

    #[test]
    fn intersect_single_edge() {
        let sized_grid = SizedGrid::new([0.0, 0.0], [5. / 4., 5. / 4.], Grid::new_from_size(4, 4));

        fn circle(x: f32, y: f32) -> f32 {
            let cx = 2.5;
            let cy = 2.5;

            let r = 15. / 8.;

            (x - cx) * (x - cx) + (y - cy) * (y - cy) - r * r
        }

        let intersection = intersect(&sized_grid, 5, &circle);
    }

    #[test]
    fn test_internal_points() {
        let sized_grid = SizedGrid::new([0.0, 0.0], [5. / 4., 5. / 4.], Grid::new_from_size(4, 4));

        fn circle(x: f32, y: f32) -> f32 {
            let cx = 2.5;
            let cy = 2.5;

            let r = 15. / 8.;

            (x - cx) * (x - cx) + (y - cy) * (y - cy) - r * r
        }

        let internal_points = internal_points(&circle, &sized_grid);

        assert_eq!(internal_points.len(), 12);
    }
}
