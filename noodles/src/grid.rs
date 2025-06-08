#[allow(non_snake_case)]
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

    #[allow(non_snake_case)]
    pub fn N(&self) -> u32 {
        self.N
    }

    #[allow(non_snake_case)]
    pub fn M(&self) -> u32 {
        self.M
    }

    #[allow(non_snake_case)]
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

#[test]
fn test_grid_ravel() {
    let g = Grid::new_from_size(2, 3);

    assert_eq!(g.ravel(1, 1), 3);
}

#[test]
fn test_cell_nodes() {
    let g = Grid::new_from_size(2, 3);

    let cell_nodes = g.cell_nodes(1, 1);
    assert_eq!(cell_nodes[0], 4);
    assert_eq!(cell_nodes[1], 5);
    assert_eq!(cell_nodes[2], 7);
    assert_eq!(cell_nodes[3], 8);
}

#[cfg(test)]
mod tests {

    use super::*;
    use assert_float_eq::assert_f32_near;
    #[test]
    fn test_cell_node_indices() {
        let g = Grid::new_from_size(2, 3);

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
        let g = Grid::new_from_size(2, 3);

        let edges = g.cell_edges(1, 1);

        assert_eq!(edges[0], 6);
        assert_eq!(edges[1], 9);
        assert_eq!(edges[2], 11);
        assert_eq!(edges[3], 8);
    }

    #[test]
    fn test_edge() {
        let g = Grid::new_from_size(6, 4);

        // Horizontal edge
        let e = g.edge(17);
        assert_eq!(e.0.0, 4);
        assert_eq!(e.0.1, 1);
        assert!(e.1 == EdgeDirection::Horizontal);

        // Horizontal edge
        let e = g.edge(13);
        assert_eq!(e.0.0, 0);
        assert_eq!(e.0.1, 1);
        assert!(e.1 == EdgeDirection::Horizontal);

        // Vertical edge
        let e = g.edge(25);
        assert_eq!(e.0.0, 6);
        assert_eq!(e.0.1, 1);
        assert!(e.1 == EdgeDirection::Vertical);

        // Vertical edge
        let e = g.edge(38);
        assert_eq!(e.0.0, 6);
        assert_eq!(e.0.1, 2);
        assert!(e.1 == EdgeDirection::Vertical);
    }

    #[test]
    fn test_edge_to_cell() {
        let g = Grid::new_from_size(6, 4);

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
}
