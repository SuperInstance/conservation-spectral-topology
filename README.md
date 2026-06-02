# conservation-spectral-topology

Conservation-Spectral-Topology (CST) unified framework for spectral graph analysis in Rust.

## Features

- **Conservation analysis**: Ratios measuring alignment of vertex attributes with spectral modes
- **Spectral decomposition**: Power iteration eigendecomposition of graph Laplacians
- **Topology**: Betti numbers, Cheeger constants, Euler characteristic
- **Spectral fingerprints**: Hash-based fingerprints for graph comparison
- **Anomaly tracking**: Sliding-window tracker for detecting anomalies in graph streams
- **Phase prediction**: Classify graph structure (tree, ring, sparse, dense, disconnected)

## Usage

```rust
use conservation_spectral_topology::*;

// Create a graph
let mut g = Graph::new(5);
for i in 0..4 {
    g.add_edge(i, i + 1, 1.0).unwrap();
}

// Full CST analysis
let report = analyze(&g);
println!("Spectral gap: {}", report.spectral_gap);
println!("Betti numbers: ({}, {})", report.betti0, report.betti1);

// Phase prediction
let phase = predict_phase(&g);
```

## Test Count

41 tests covering all major functionality.

## License

MIT
