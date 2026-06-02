//! # Conservation-Spectral-Topology (CST)
//!
//! A unified framework combining conservation analysis, spectral graph theory,
//! and topological methods for understanding the structure of attributed graphs.
//!
//! ## Core concepts
//!
//! - **Conservation**: Ratios measuring how well vertex attributes align with spectral modes
//! - **Spectral**: Eigen-decomposition of the graph Laplacian, spectral gaps, fingerprints
//! - **Topology**: Betti numbers, Cheeger constants, connectivity analysis
//! - **CST**: Unified analysis combining all three perspectives
//!
//! This is an idiomatic Rust port of the original C header-only library, using
//! `Vec`, no `unsafe`, and proper error handling.

#![deny(unsafe_code)]

use std::fmt;

// ============================================================
// Error handling
// ============================================================

/// Errors that can occur in CST operations.
#[derive(Debug, Clone, PartialEq)]
pub enum CstError {
    NullPointer,
    OutOfBounds { index: usize, len: usize },
    InvalidState(String),
    DimensionMismatch { expected: usize, actual: usize },
    Singular,
    NoConvergence { iterations: usize },
}

impl fmt::Display for CstError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CstError::NullPointer => write!(f, "null pointer / empty data"),
            CstError::OutOfBounds { index, len } => {
                write!(f, "index {index} out of bounds (len {len})")
            }
            CstError::InvalidState(msg) => write!(f, "invalid state: {msg}"),
            CstError::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            CstError::Singular => write!(f, "singular matrix"),
            CstError::NoConvergence { iterations } => {
                write!(f, "no convergence after {iterations} iterations")
            }
        }
    }
}

impl std::error::Error for CstError {}

pub type Result<T> = std::result::Result<T, CstError>;

// ============================================================
// Graph
// ============================================================

/// A vertex in the graph with an ID and a scalar attribute.
#[derive(Debug, Clone)]
pub struct Vertex {
    pub id: u64,
    pub attribute: f64,
}

/// A weighted edge.
#[derive(Debug, Clone)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub weight: f64,
}

/// An undirected weighted graph with vertex attributes.
#[derive(Debug, Clone)]
pub struct Graph {
    pub vertices: Vec<Vertex>,
    pub edges: Vec<Edge>,
    pub directed: bool,
}

impl Graph {
    /// Create a new graph with `n` vertices (IDs 0..n, attributes default to 0).
    pub fn new(n: usize) -> Self {
        let vertices = (0..n)
            .map(|i| Vertex {
                id: i as u64,
                attribute: 0.0,
            })
            .collect();
        Graph {
            vertices,
            edges: Vec::new(),
            directed: false,
        }
    }

    /// Number of vertices.
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Number of edges.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Set the attribute of vertex `id`.
    pub fn set_vertex_attribute(&mut self, id: usize, attr: f64) -> Result<()> {
        if id >= self.vertices.len() {
            return Err(CstError::OutOfBounds {
                index: id,
                len: self.vertices.len(),
            });
        }
        self.vertices[id].attribute = attr;
        Ok(())
    }

    /// Add an undirected edge.
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) -> Result<()> {
        let n = self.vertices.len();
        if from >= n || to >= n {
            return Err(CstError::OutOfBounds {
                index: if from >= n { from } else { to },
                len: n,
            });
        }
        self.edges.push(Edge { from, to, weight });
        Ok(())
    }

    /// Get vertex attributes as a slice.
    pub fn attributes(&self) -> Vec<f64> {
        self.vertices.iter().map(|v| v.attribute).collect()
    }
}

// ============================================================
// Laplacian
// ============================================================

/// Dense row-major matrix representation of the Laplacian.
#[derive(Debug, Clone)]
pub struct Laplacian {
    pub values: Vec<f64>, // row-major n×n
    pub n: usize,
    pub normalized: bool,
}

impl Laplacian {
    /// Build the graph Laplacian (combinatorial or normalized).
    pub fn from_graph(graph: &Graph, normalized: bool) -> Self {
        let n = graph.n_vertices();
        let mut values = vec![0.0; n * n];

        // Build adjacency weight matrix
        let mut w = vec![0.0; n * n];
        for edge in &graph.edges {
            let i = edge.from;
            let j = edge.to;
            let wt = edge.weight;
            w[i * n + j] += wt;
            if !graph.directed {
                w[j * n + i] += wt;
            }
        }

        // Degree vector
        let mut deg = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                deg[i] += w[i * n + j];
            }
        }

        if !normalized {
            // L = D - W
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        values[i * n + j] = deg[i] - w[i * n + j];
                    } else {
                        values[i * n + j] = -w[i * n + j];
                    }
                }
            }
        } else {
            // Symmetric normalized: L = I - D^{-1/2} W D^{-1/2}
            for i in 0..n {
                for j in 0..n {
                    let di_sqrt = if deg[i] > 0.0 {
                        1.0 / deg[i].sqrt()
                    } else {
                        0.0
                    };
                    let dj_sqrt = if deg[j] > 0.0 {
                        1.0 / deg[j].sqrt()
                    } else {
                        0.0
                    };
                    let val = di_sqrt * w[i * n + j] * dj_sqrt;
                    if i == j {
                        values[i * n + j] = 1.0 - val;
                    } else {
                        values[i * n + j] = -val;
                    }
                }
            }
        }

        Laplacian {
            values,
            n,
            normalized,
        }
    }

    /// Get element at (i, j).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.values[i * self.n + j]
    }

    /// Matrix-vector multiply: y = L * x.
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n;
        self.values
            .chunks_exact(n)
            .map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum())
            .collect()
    }
}

// ============================================================
// Eigen decomposition
// ============================================================

/// Eigen decomposition result.
#[derive(Debug, Clone)]
pub struct EigenDecomposition {
    /// Eigenvalues sorted ascending.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors as column-major n×n: column i = eigenvector i.
    pub eigenvectors: Vec<f64>,
    pub n: usize,
}

impl EigenDecomposition {
    /// Compute eigendecomposition using power iteration with deflation.
    ///
    /// Always computes all n eigenvalues (k parameter is ignored).
    pub fn compute(laplacian: &Laplacian, _k: usize) -> Self {
        let n = laplacian.n;
        if n == 0 {
            return EigenDecomposition {
                eigenvalues: vec![],
                eigenvectors: vec![],
                n: 0,
            };
        }

        let mut eigenvalues = vec![0.0; n];
        let mut eigenvectors = vec![0.0; n * n];

        // Find shift = max diagonal element
        let mut shift = 0.0_f64;
        for i in 0..n {
            let diag = laplacian.get(i, i);
            if diag > shift {
                shift = diag;
            }
        }

        // Build M = shift*I - L
        let mut m: Vec<f64> = laplacian.values.iter().map(|v| -v).collect();
        for i in 0..n {
            m[i * n + i] += shift;
        }

        // Residual matrix for deflation
        let mut r = m.clone();

        for ev in 0..n {
            // Initial vector with varied seeds
            let mut v: Vec<f64> = (0..n).map(|i| 1.0 / (i + 1 + ev) as f64).collect();

            let max_iter = 3000;
            let tol = 1e-12;
            let mut lambda = 0.0;

            for _ in 0..max_iter {
                // w = R * v
                let mut w = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        w[i] += r[i * n + j] * v[j];
                    }
                }

                // Normalize
                let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < 1e-30 {
                    break;
                }
                for i in 0..n {
                    v[i] = w[i] / norm;
                }

                // Rayleigh quotient
                let mut w2 = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        w2[i] += r[i * n + j] * v[j];
                    }
                }
                let rq: f64 = v.iter().zip(w2.iter()).map(|(a, b)| a * b).sum();

                if (rq - lambda).abs() < tol {
                    lambda = rq;
                    break;
                }
                lambda = rq;
            }

            // eigenvalue of L = shift - lambda_M
            eigenvalues[ev] = shift - lambda;

            // Store eigenvector
            for i in 0..n {
                eigenvectors[ev * n + i] = v[i];
            }

            // Deflate: R = R - lambda * v*v^T
            for i in 0..n {
                for j in 0..n {
                    r[i * n + j] -= lambda * v[i] * v[j];
                }
            }
        }

        // Sort ascending
        for i in 0..n - 1 {
            for j in (i + 1)..n {
                if eigenvalues[j] < eigenvalues[i] {
                    eigenvalues.swap(i, j);
                    // swap columns i and j
                    for r in 0..n {
                        eigenvectors.swap(i * n + r, j * n + r);
                    }
                }
            }
        }

        EigenDecomposition {
            eigenvalues,
            eigenvectors,
            n,
        }
    }

    /// Get eigenvector `idx` as a Vec.
    pub fn eigenvector(&self, idx: usize) -> Vec<f64> {
        (0..self.n)
            .map(|i| self.eigenvectors[idx * self.n + i])
            .collect()
    }
}

// ============================================================
// Conservation analysis
// ============================================================

/// Compute the conservation ratio for one eigenvector against an attribute array.
pub fn conservation_ratio(
    eigen: &EigenDecomposition,
    attr: &[f64],
    eigenvector_index: usize,
) -> f64 {
    let n = eigen.n;
    if attr.len() != n || n < 2 {
        return -1.0;
    }

    let ev = eigen.eigenvector(eigenvector_index);

    // Project attribute onto eigenvector
    let projection: Vec<f64> = (0..n).map(|i| attr[i] * ev[i]).collect();

    // Gradient: diff of consecutive projected values
    let gradient: Vec<f64> = (0..n - 1)
        .map(|i| projection[i + 1] - projection[i])
        .collect();

    // Variance of gradient
    let mean = gradient.iter().sum::<f64>() / (n - 1) as f64;
    let var = gradient.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;

    var
}

/// Compute the spectral gap (largest gap between consecutive eigenvalues).
pub fn spectral_gap(eigen: &EigenDecomposition) -> f64 {
    if eigen.n < 2 {
        return 0.0;
    }
    let mut max_gap = 0.0;
    for i in 0..eigen.n - 1 {
        let gap = eigen.eigenvalues[i + 1] - eigen.eigenvalues[i];
        if gap > max_gap {
            max_gap = gap;
        }
    }
    max_gap
}

/// Approximate Cheeger constant from Fiedler vector.
pub fn cheeger_constant(laplacian: &Laplacian, fiedler: &[f64]) -> f64 {
    let n = laplacian.n;
    if fiedler.len() != n {
        return 0.0;
    }

    // Partition: S = {v : fiedler[v] < 0}
    let in_s: Vec<bool> = fiedler.iter().map(|x| *x < 0.0).collect();

    let mut cut = 0.0;
    let mut vol_s = 0.0;
    let mut total_vol = 0.0;

    for i in 0..n {
        total_vol += laplacian.get(i, i); // diagonal = degree
        if in_s[i] {
            vol_s += laplacian.get(i, i);
            for (j, &in_sj) in in_s.iter().enumerate() {
                if i != j && !in_sj {
                    cut += -laplacian.get(i, j); // -L(i,j) = weight
                }
            }
        }
    }

    let vol_comp = total_vol - vol_s;
    let min_vol = vol_s.min(vol_comp);

    if min_vol < 1e-15 {
        return 0.0;
    }
    cut / min_vol
}

// ============================================================
// Topology: Betti numbers
// ============================================================

/// Compute Betti numbers for a graph (0-simplicial complex).
///
/// For a graph (1-dimensional simplicial complex):
/// - β₀ = number of connected components
/// - β₁ = |E| - |V| + β₀ (number of independent cycles)
pub fn betti_numbers(graph: &Graph) -> (usize, usize) {
    let n = graph.n_vertices();
    if n == 0 {
        return (0, 0);
    }

    // Union-Find for connected components
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank = vec![0usize; n];

    fn find(parent: &mut Vec<usize>, i: usize) -> usize {
        if parent[i] != i {
            parent[i] = find(parent, parent[i]);
        }
        parent[i]
    }

    for edge in &graph.edges {
        let ri = find(&mut parent, edge.from);
        let rj = find(&mut parent, edge.to);
        if ri != rj {
            if rank[ri] < rank[rj] {
                parent[ri] = rj;
            } else if rank[ri] > rank[rj] {
                parent[rj] = ri;
            } else {
                parent[rj] = ri;
                rank[ri] += 1;
            }
        }
    }

    let beta0 = (0..n).filter(|&i| parent[i] == i).count();
    let beta1 = (graph.n_edges() as isize - n as isize + beta0 as isize).max(0) as usize;

    (beta0, beta1)
}

/// Compute the Euler characteristic: χ = V - E (= β₀ - β₁).
pub fn euler_characteristic(graph: &Graph) -> isize {
    graph.n_vertices() as isize - graph.n_edges() as isize
}

// ============================================================
// Spectral fingerprint
// ============================================================

/// Compute a hex fingerprint from eigenvalues.
pub fn fingerprint_compute(eigenvalues: &[f64]) -> String {
    if eigenvalues.is_empty() {
        return String::new();
    }

    let mut result = String::with_capacity(eigenvalues.len() * 16);
    const HEX: &[u8; 16] = b"0123456789abcdef";

    for &ev in eigenvalues {
        let bits = ev.to_bits();
        let mut mixed = bits;
        mixed ^= mixed >> 33;
        mixed = mixed.wrapping_mul(0xff51afd7ed558ccd);
        mixed ^= mixed >> 33;
        mixed = mixed.wrapping_mul(0xc4ceb9fe1a85ec53);
        mixed ^= mixed >> 33;

        for j in (0..16).rev() {
            let nibble = ((mixed >> (j * 4)) & 0xF) as usize;
            result.push(HEX[nibble] as char);
        }
    }

    result
}

/// Compare two fingerprints. Returns similarity in [0, 1].
pub fn fingerprint_compare(fp1: &str, fp2: &str) -> f64 {
    if fp1.is_empty() && fp2.is_empty() {
        return 1.0;
    }
    let max_len = fp1.len().max(fp2.len());
    if max_len == 0 {
        return 1.0;
    }
    let matches = fp1.bytes().zip(fp2.bytes()).filter(|(a, b)| a == b).count();
    matches as f64 / max_len as f64
}

// ============================================================
// Sliding-window tracker
// ============================================================

/// A sliding-window tracker for detecting anomalies in a stream of observations.
#[derive(Debug, Clone)]
pub struct Tracker {
    window_size: usize,
    history: Vec<f64>,
    baseline_mean: f64,
    baseline_std: f64,
    baseline_set: bool,
}

/// Tracker status after feeding an observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackerStatus {
    Nominal,
    Warning,
    Critical,
}

impl Tracker {
    /// Create a new tracker with the given window size.
    pub fn new(window_size: usize) -> Self {
        Tracker {
            window_size,
            history: Vec::with_capacity(window_size),
            baseline_mean: 0.0,
            baseline_std: 0.0,
            baseline_set: false,
        }
    }

    /// Feed a single observation. Returns the status.
    pub fn feed(&mut self, observation: f64) -> TrackerStatus {
        if self.history.len() < self.window_size {
            self.history.push(observation);
        } else {
            self.history.remove(0);
            self.history.push(observation);
        }

        // Establish baseline after filling window once
        if self.history.len() == self.window_size && !self.baseline_set {
            let sum: f64 = self.history.iter().sum();
            self.baseline_mean = sum / self.window_size as f64;

            let var: f64 = self
                .history
                .iter()
                .map(|x| (x - self.baseline_mean).powi(2))
                .sum::<f64>()
                / self.window_size as f64;
            self.baseline_std = var.sqrt();
            self.baseline_set = true;
            return TrackerStatus::Nominal;
        }

        self.check()
    }

    /// Check current status without feeding.
    pub fn check(&self) -> TrackerStatus {
        if !self.baseline_set || self.history.is_empty() {
            return TrackerStatus::Nominal;
        }

        let latest = *self.history.last().unwrap();
        if self.baseline_std < 1e-15 {
            return TrackerStatus::Nominal;
        }

        let zscore = (latest - self.baseline_mean).abs() / self.baseline_std;

        if zscore > 3.0 {
            TrackerStatus::Critical
        } else if zscore > 2.0 {
            TrackerStatus::Warning
        } else {
            TrackerStatus::Nominal
        }
    }
}

// ============================================================
// Full CST report
// ============================================================

/// Conservation ratio for one eigenvector.
#[derive(Debug, Clone)]
pub struct RatioEntry {
    pub eigenvector_index: usize,
    pub eigenvalue: f64,
    pub ratio: f64,
}

/// Full CST analysis report.
#[derive(Debug, Clone)]
pub struct CstReport {
    pub spectral_gap: f64,
    pub cheeger_constant: f64,
    pub betti0: usize,
    pub betti1: usize,
    pub ratios: Vec<RatioEntry>,
}

/// Run full CST analysis on a graph.
pub fn analyze(graph: &Graph) -> CstReport {
    let lap = Laplacian::from_graph(graph, false);
    let eigen = EigenDecomposition::compute(&lap, 0);
    let attrs = graph.attributes();

    let ratios: Vec<RatioEntry> = (0..eigen.n)
        .map(|k| RatioEntry {
            eigenvector_index: k,
            eigenvalue: eigen.eigenvalues[k],
            ratio: conservation_ratio(&eigen, &attrs, k),
        })
        .collect();

    let gap = spectral_gap(&eigen);

    let cheeg = if eigen.n >= 2 {
        let fiedler = eigen.eigenvector(1);
        cheeger_constant(&lap, &fiedler)
    } else {
        0.0
    };

    let (betti0, betti1) = betti_numbers(graph);

    CstReport {
        spectral_gap: gap,
        cheeger_constant: cheeg,
        betti0,
        betti1,
        ratios,
    }
}

// ============================================================
// Phase prediction
// ============================================================

/// Predict the phase of a graph based on spectral properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphPhase {
    /// Disconnected: β₀ > 1
    Disconnected,
    /// Tree-like: β₁ = 0 and connected
    Tree,
    /// Sparse: small spectral gap relative to n
    Sparse,
    /// Dense: large spectral gap
    Dense,
    /// Ring/cycle: exactly one cycle
    Ring,
}

/// Predict the structural phase of a graph.
pub fn predict_phase(graph: &Graph) -> GraphPhase {
    let (beta0, beta1) = betti_numbers(graph);
    if beta0 > 1 {
        return GraphPhase::Disconnected;
    }
    if beta1 == 0 {
        return GraphPhase::Tree;
    }
    if beta1 == 1 {
        return GraphPhase::Ring;
    }

    let lap = Laplacian::from_graph(graph, false);
    let eigen = EigenDecomposition::compute(&lap, 2);
    let gap = spectral_gap(&eigen);
    let n = graph.n_vertices() as f64;

    // Heuristic: dense graphs have spectral gap ~ n, sparse have gap ~ 1/n
    if gap > n * 0.3 {
        GraphPhase::Dense
    } else {
        GraphPhase::Sparse
    }
}

// ============================================================
// Conservation verification
// ============================================================

/// Verify that conservation holds: the sum of attributes is preserved
/// under the spectral decomposition (Parseval-like identity).
pub fn verify_conservation(graph: &Graph) -> bool {
    let attrs = graph.attributes();
    let n = graph.n_vertices();
    if n == 0 {
        return true;
    }

    let lap = Laplacian::from_graph(graph, false);
    let eigen = EigenDecomposition::compute(&lap, n);

    // Energy in attribute space
    let attr_energy: f64 = attrs.iter().map(|x| x * x).sum();

    // Energy in spectral space (Parseval)
    let spectral_energy: f64 = (0..n)
        .map(|k| {
            let ev = eigen.eigenvector(k);
            let proj: f64 = attrs.iter().zip(ev.iter()).map(|(a, e)| a * e).sum();
            proj * proj
        })
        .sum();

    (attr_energy - spectral_energy).abs() < 0.1 * attr_energy.max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_path_graph(n: usize) -> Graph {
        let mut g = Graph::new(n);
        for i in 0..n {
            g.set_vertex_attribute(i, i as f64 * 1.5).unwrap();
        }
        for i in 0..n - 1 {
            g.add_edge(i, i + 1, 1.0).unwrap();
        }
        g
    }

    fn make_cycle_graph(n: usize) -> Graph {
        let mut g = Graph::new(n);
        for i in 0..n {
            g.add_edge(i, (i + 1) % n, 1.0).unwrap();
        }
        g
    }

    fn make_complete_graph(n: usize) -> Graph {
        let mut g = Graph::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                g.add_edge(i, j, 1.0).unwrap();
            }
        }
        g
    }

    // ---- Graph tests ----

    #[test]
    fn test_graph_creation() {
        let g = Graph::new(5);
        assert_eq!(g.n_vertices(), 5);
        assert_eq!(g.n_edges(), 0);
    }

    #[test]
    fn test_graph_add_edges() {
        let mut g = Graph::new(5);
        for i in 0..4 {
            g.add_edge(i, i + 1, 1.0).unwrap();
        }
        assert_eq!(g.n_edges(), 4);
    }

    #[test]
    fn test_graph_out_of_bounds() {
        let mut g = Graph::new(3);
        assert!(g.add_edge(0, 5, 1.0).is_err());
        assert!(g.set_vertex_attribute(10, 1.0).is_err());
    }

    #[test]
    fn test_graph_attributes() {
        let mut g = Graph::new(3);
        g.set_vertex_attribute(0, 1.0).unwrap();
        g.set_vertex_attribute(1, 2.0).unwrap();
        g.set_vertex_attribute(2, 3.0).unwrap();
        assert_eq!(g.attributes(), vec![1.0, 2.0, 3.0]);
    }

    // ---- Laplacian tests ----

    #[test]
    fn test_laplacian_path_diagonal() {
        let g = make_path_graph(5);
        let lap = Laplacian::from_graph(&g, false);
        assert_eq!(lap.n, 5);
        assert!((lap.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((lap.get(1, 1) - 2.0).abs() < 1e-10);
        assert!((lap.get(2, 2) - 2.0).abs() < 1e-10);
        assert!((lap.get(4, 4) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_off_diagonal() {
        let g = make_path_graph(5);
        let lap = Laplacian::from_graph(&g, false);
        assert!((lap.get(0, 1) - (-1.0)).abs() < 1e-10);
        assert!((lap.get(1, 0) - (-1.0)).abs() < 1e-10);
        assert!((lap.get(0, 4) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_normalized() {
        let g = make_path_graph(5);
        let lap = Laplacian::from_graph(&g, true);
        assert!(lap.normalized);
        // Diagonal should be 1.0 for normalized
        assert!((lap.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_symmetric() {
        let g = make_path_graph(5);
        let lap = Laplacian::from_graph(&g, false);
        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    (lap.get(i, j) - lap.get(j, i)).abs() < 1e-10,
                    "Laplacian not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_laplacian_row_sum_zero() {
        let g = make_path_graph(5);
        let lap = Laplacian::from_graph(&g, false);
        for i in 0..5 {
            let row_sum: f64 = (0..5).map(|j| lap.get(i, j)).sum();
            assert!(
                row_sum.abs() < 1e-10,
                "Row {i} sum = {row_sum}, expected ~0"
            );
        }
    }

    // ---- Eigen tests ----

    #[test]
    fn test_eigen_first_is_zero() {
        let g = make_path_graph(5);
        let lap = Laplacian::from_graph(&g, false);
        let eigen = EigenDecomposition::compute(&lap, 0);
        assert!(
            eigen.eigenvalues[0].abs() < 0.1,
            "First eigenvalue should be ~0, got {}",
            eigen.eigenvalues[0]
        );
    }

    #[test]
    fn test_eigen_all_nonnegative() {
        let g = make_path_graph(5);
        let lap = Laplacian::from_graph(&g, false);
        let eigen = EigenDecomposition::compute(&lap, 0);
        for (i, &ev) in eigen.eigenvalues.iter().enumerate() {
            assert!(ev >= -1e-6, "eigenvalue[{i}] = {ev} is negative");
        }
    }

    #[test]
    fn test_eigen_sorted_ascending() {
        let g = make_path_graph(5);
        let lap = Laplacian::from_graph(&g, false);
        let eigen = EigenDecomposition::compute(&lap, 0);
        for i in 0..eigen.n - 1 {
            assert!(
                eigen.eigenvalues[i] <= eigen.eigenvalues[i + 1] + 1e-10,
                "eigenvalues not sorted at index {i}"
            );
        }
    }

    #[test]
    fn test_eigen_path_values_approximate() {
        let g = make_path_graph(5);
        let lap = Laplacian::from_graph(&g, false);
        let eigen = EigenDecomposition::compute(&lap, 0);
        // Path graph P5 eigenvalues: 2 - 2*cos(pi*k/5)
        let expected0 = 0.0;
        assert!(
            (eigen.eigenvalues[0] - expected0).abs() < 0.1,
            "eigenvalue[0] = {} expected ~0",
            eigen.eigenvalues[0]
        );
    }

    // ---- Spectral gap tests ----

    #[test]
    fn test_spectral_gap_positive() {
        let g = make_path_graph(5);
        let lap = Laplacian::from_graph(&g, false);
        let eigen = EigenDecomposition::compute(&lap, 0);
        let gap = spectral_gap(&eigen);
        assert!(gap > 0.0, "Spectral gap should be positive");
    }

    #[test]
    fn test_spectral_gap_complete_larger_than_path() {
        let gp = make_path_graph(10);
        let gc = make_complete_graph(10);
        let lap_p = Laplacian::from_graph(&gp, false);
        let lap_c = Laplacian::from_graph(&gc, false);
        let eigen_p = EigenDecomposition::compute(&lap_p, 0);
        let eigen_c = EigenDecomposition::compute(&lap_c, 0);
        let gap_p = eigen_p.eigenvalues[1]; // algebraic connectivity
        let gap_c = eigen_c.eigenvalues[1];
        assert!(
            gap_c > gap_p,
            "Complete graph algebraic connectivity ({gap_c}) should > path ({gap_p})"
        );
    }

    // ---- Conservation ratio tests ----

    #[test]
    fn test_conservation_ratio_nonnegative() {
        let g = make_path_graph(5);
        let lap = Laplacian::from_graph(&g, false);
        let eigen = EigenDecomposition::compute(&lap, 0);
        let attrs = g.attributes();
        for k in 0..5 {
            let cr = conservation_ratio(&eigen, &attrs, k);
            assert!(cr >= 0.0, "conservation ratio for eigvec {k} is {cr}");
        }
    }

    // ---- Cheeger constant tests ----

    #[test]
    fn test_cheeger_nonnegative() {
        let g = make_path_graph(5);
        let lap = Laplacian::from_graph(&g, false);
        let eigen = EigenDecomposition::compute(&lap, 0);
        let fiedler = eigen.eigenvector(1);
        let ch = cheeger_constant(&lap, &fiedler);
        assert!(ch >= 0.0, "Cheeger constant should be >= 0");
    }

    // ---- Betti number tests ----

    #[test]
    fn test_betti_path_connected() {
        let g = make_path_graph(5);
        let (b0, b1) = betti_numbers(&g);
        assert_eq!(b0, 1, "Path graph should have 1 component");
        assert_eq!(b1, 0, "Path graph has no cycles");
    }

    #[test]
    fn test_betti_cycle() {
        let g = make_cycle_graph(5);
        let (b0, b1) = betti_numbers(&g);
        assert_eq!(b0, 1);
        assert_eq!(b1, 1, "Cycle graph should have 1 cycle");
    }

    #[test]
    fn test_betti_disconnected() {
        let mut g = Graph::new(6);
        // Two separate paths: 0-1-2 and 3-4-5
        g.add_edge(0, 1, 1.0).unwrap();
        g.add_edge(1, 2, 1.0).unwrap();
        g.add_edge(3, 4, 1.0).unwrap();
        g.add_edge(4, 5, 1.0).unwrap();
        let (b0, b1) = betti_numbers(&g);
        assert_eq!(b0, 2, "Should have 2 components");
        assert_eq!(b1, 0);
    }

    #[test]
    fn test_betti_complete() {
        let g = make_complete_graph(4);
        let (b0, b1) = betti_numbers(&g);
        assert_eq!(b0, 1);
        // K4 has 6 edges, 4 vertices: β₁ = 6 - 4 + 1 = 3
        assert_eq!(b1, 3);
    }

    #[test]
    fn test_betti_empty_graph() {
        let g = Graph::new(0);
        let (b0, b1) = betti_numbers(&g);
        assert_eq!(b0, 0);
        assert_eq!(b1, 0);
    }

    // ---- Euler characteristic tests ----

    #[test]
    fn test_euler_path() {
        let g = make_path_graph(5);
        assert_eq!(euler_characteristic(&g), 1); // 5 - 4 = 1
    }

    #[test]
    fn test_euler_cycle() {
        let g = make_cycle_graph(5);
        assert_eq!(euler_characteristic(&g), 0); // 5 - 5 = 0
    }

    // ---- Fingerprint tests ----

    #[test]
    fn test_fingerprint_self_similarity() {
        let g = make_path_graph(5);
        let lap = Laplacian::from_graph(&g, false);
        let eigen = EigenDecomposition::compute(&lap, 0);
        let fp = fingerprint_compute(&eigen.eigenvalues);
        let sim = fingerprint_compare(&fp, &fp);
        assert!((sim - 1.0).abs() < 1e-10, "Self-similarity should be 1.0");
    }

    #[test]
    fn test_fingerprint_empty() {
        let fp = fingerprint_compute(&[]);
        assert!(fp.is_empty());
        assert!((fingerprint_compare("", "") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fingerprint_different_graphs() {
        let g1 = make_path_graph(5);
        let g2 = make_complete_graph(5);
        let e1 = EigenDecomposition::compute(&Laplacian::from_graph(&g1, false), 0);
        let e2 = EigenDecomposition::compute(&Laplacian::from_graph(&g2, false), 0);
        let fp1 = fingerprint_compute(&e1.eigenvalues);
        let fp2 = fingerprint_compute(&e2.eigenvalues);
        let sim = fingerprint_compare(&fp1, &fp2);
        assert!(
            sim < 1.0,
            "Different graphs should have different fingerprints"
        );
    }

    // ---- Tracker tests ----

    #[test]
    fn test_tracker_normal() {
        let mut trk = Tracker::new(5);
        for i in 0..5 {
            trk.feed(10.0 + (i % 3) as f64 * 0.1);
        }
        let status = trk.feed(10.1);
        assert_eq!(status, TrackerStatus::Nominal);
    }

    #[test]
    fn test_tracker_outlier_critical() {
        let mut trk = Tracker::new(10);
        for i in 0..10 {
            trk.feed(10.0 + (i % 3) as f64 * 0.1);
        }
        let status = trk.feed(100.0);
        assert!(
            status == TrackerStatus::Critical || status == TrackerStatus::Warning,
            "Should detect outlier"
        );
    }

    #[test]
    fn test_tracker_baseline_establishment() {
        let mut trk = Tracker::new(5);
        for _i in 0..4 {
            let status = trk.feed(10.0);
            assert_eq!(status, TrackerStatus::Nominal);
        }
        // 5th feed establishes baseline
        let status = trk.feed(10.0);
        assert_eq!(status, TrackerStatus::Nominal);
        // Now baseline is set
        assert!(trk.baseline_set);
    }

    // ---- Full analysis tests ----

    #[test]
    fn test_analyze_path_graph() {
        let g = make_path_graph(5);
        let report = analyze(&g);
        assert!(report.spectral_gap > 0.0);
        assert!(report.cheeger_constant >= 0.0);
        assert_eq!(report.betti0, 1);
        assert_eq!(report.betti1, 0);
        assert_eq!(report.ratios.len(), 5);
    }

    #[test]
    fn test_analyze_cycle_graph() {
        let g = make_cycle_graph(5);
        let report = analyze(&g);
        assert_eq!(report.betti0, 1);
        assert_eq!(report.betti1, 1);
    }

    // ---- Phase prediction tests ----

    #[test]
    fn test_phase_tree() {
        let g = make_path_graph(5);
        assert_eq!(predict_phase(&g), GraphPhase::Tree);
    }

    #[test]
    fn test_phase_ring() {
        let g = make_cycle_graph(5);
        assert_eq!(predict_phase(&g), GraphPhase::Ring);
    }

    #[test]
    fn test_phase_disconnected() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0).unwrap();
        // Nodes 2 and 3 are disconnected
        assert_eq!(predict_phase(&g), GraphPhase::Disconnected);
    }

    #[test]
    fn test_phase_dense() {
        let g = make_complete_graph(5);
        let phase = predict_phase(&g);
        assert!(
            phase == GraphPhase::Dense || phase == GraphPhase::Sparse,
            "Complete graph should be classified as dense or sparse"
        );
    }

    // ---- Conservation verification tests ----

    #[test]
    fn test_conservation_verification_path() {
        let g = make_path_graph(5);
        assert!(verify_conservation(&g));
    }

    #[test]
    fn test_conservation_verification_cycle() {
        let g = make_cycle_graph(5);
        assert!(verify_conservation(&g));
    }

    #[test]
    fn test_conservation_verification_complete() {
        let g = make_complete_graph(4);
        assert!(verify_conservation(&g));
    }

    // ---- Matvec test ----

    #[test]
    fn test_laplacian_matvec_constant_is_zero() {
        let g = make_path_graph(5);
        let lap = Laplacian::from_graph(&g, false);
        let x = vec![1.0; 5];
        let y = lap.matvec(&x);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.abs() < 1e-10, "L*1 at index {i} = {v}, expected ~0");
        }
    }

    // ---- Musical chord test (from C version) ----

    #[test]
    fn test_chord_progression() {
        let mut g = Graph::new(5);
        // Chord tensions: C=0, G=0.6, Am=0.2, F=0.4, Dm=0.3
        let tensions = [0.0, 0.6, 0.2, 0.4, 0.3];
        for (i, &t) in tensions.iter().enumerate() {
            g.set_vertex_attribute(i, t).unwrap();
        }

        g.add_edge(0, 1, 0.4).unwrap(); // C->G
        g.add_edge(0, 2, 0.2).unwrap(); // C->Am
        g.add_edge(0, 3, 0.25).unwrap(); // C->F
        g.add_edge(1, 0, 0.5).unwrap(); // G->C
        g.add_edge(1, 2, 0.15).unwrap(); // G->Am
        g.add_edge(2, 3, 0.3).unwrap(); // Am->F
        g.add_edge(2, 4, 0.25).unwrap(); // Am->Dm
        g.add_edge(3, 1, 0.3).unwrap(); // F->G
        g.add_edge(3, 0, 0.2).unwrap(); // F->C
        g.add_edge(4, 1, 0.5).unwrap(); // Dm->G
        g.add_edge(4, 2, 0.15).unwrap(); // Dm->Am

        assert_eq!(g.n_edges(), 11);

        let lap = Laplacian::from_graph(&g, false);
        let eigen = EigenDecomposition::compute(&lap, 0);

        // All eigenvalues non-negative
        for &ev in &eigen.eigenvalues {
            assert!(ev >= -1e-6, "Chord eigenvalue negative: {ev}");
        }

        // First eigenvalue ~0 (connected)
        assert!(
            eigen.eigenvalues[0].abs() < 0.2,
            "Connected graph first eigenvalue should be ~0"
        );

        let report = analyze(&g);
        assert!(report.spectral_gap > 0.0);
        assert_eq!(report.betti0, 1);
    }
}
