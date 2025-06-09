use std::collections::HashMap;

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

use crate::{
    grid::{EdgeDirection, SizedGrid},
    tessellation,
};

pub fn case<F>(sized_grid: &SizedGrid, i: u32, j: u32, func: &F) -> u32
where
    F: Fn(f32, f32) -> f32,
{
    let nodes = sized_grid.grid().cell_node_indices(i, j);

    let points = nodes.map(|(i, j)| sized_grid.point(i, j));

    let states = points.map(|point| func(point[0], point[1]) < f32::EPSILON);

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

pub fn topo_segment<F>(f: &F, grid: &SizedGrid) -> Vec<((u32, u32), (u32, u32))>
where
    F: Fn(f32, f32) -> f32,
{
    #[allow(non_snake_case)]
    let N = grid.grid().N();

    #[allow(non_snake_case)]
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
    segments
}

pub fn internal_points<F>(f: &F, grid: &SizedGrid) -> HashMap<(u32, u32), (f32, f32)>
where
    F: Fn(f32, f32) -> f32,
{
    let mut res: HashMap<(u32, u32), (f32, f32)> = Default::default();

    #[allow(non_snake_case)]
    let N = grid.grid().N();
    #[allow(non_snake_case)]
    let M = grid.grid().M();

    for cell_j in 0..M {
        for cell_i in 0..N {
            let cell_index = cell_j * N + cell_i;
            let case = case(&grid, cell_i, cell_j, &f);

            let cell_edges = grid.grid().cell_edges(cell_i, cell_j);

            let edge_to_internal = LOCAL_EDGE_TO_INTERNAL_POINT[case as usize];
            let n_internal_points = edge_to_internal.iter().max().unwrap();
            let mut internal_points: Vec<(f32, f32)> = vec![(0., 0.); *n_internal_points as usize];
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
    let segments = topo_segment(&f, &grid);

    // Create internal points
    let internal_points = internal_points(&f, &grid);

    let mut res = crate::tessellation::Contour::default();

    let mut internal_point_to_vertex: HashMap<(u32, u32), u32> = Default::default();

    for internal_point in internal_points {
        let (ids, coords) = internal_point;
        res.points.push(crate::tessellation::Point2D {
            x: coords.0,
            y: coords.1,
        });
        internal_point_to_vertex.insert(ids, (res.points.len() - 1) as u32);
    }

    for segment in segments {
        let start_vertex = match internal_point_to_vertex.get(&segment.0) {
            Some(value) => value,
            None => {
                let (i, j) = grid.grid().unravel(segment.0.0);
                let case = case(&grid, i, j, &f);
                panic!(
                    "Start vertex malformed in {:?}, {:?}, {:?}",
                    segment,
                    (i, j),
                    case
                );
            }
        };

        let end_vertex = internal_point_to_vertex.get(&segment.1).unwrap();
        res.edges
            .push((*start_vertex as usize, *end_vertex as usize));
    }

    res
}

pub fn contour_formula(
    formula: &str,
    grid: SizedGrid,
) -> Result<tessellation::Contour, meval::Error> {
    let f64_func: meval::Expr = formula.parse()?;
    let f64_func = f64_func.bind2("x", "y")?;

    let f32_func = |x: f32, y: f32| f64_func(x as f64, y as f64) as f32;

    let tessellation = contour(f32_func, grid);

    Ok(tessellation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::grid::SizedGrid;
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

        let segments = topo_segment(&circle, &sized_grid);

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

        let _intersection = intersect(&sized_grid, 5, &circle);
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
