# 3D Gaussian Splatting Viewer

A real-time 3D Gaussian Splatting viewer built on the [Nightshade](https://github.com/matthewjberger/nightshade) game engine. Loads pre-trained `.ply` files from the standard [3DGS training pipeline](https://github.com/graphdeco-inria/gaussian-splatting) and renders them using GPU compute preprocessing, bitonic sort, and alpha-blended instanced quad rendering.

![ezgif-10143c020bdfff9a](https://github.com/user-attachments/assets/9d4b76b2-e91f-4259-be9a-18669af60452)

## Usage

```bash
cargo run --release -- <path_to.ply>
```

The viewer expects a `.ply` file output from the 3DGS training pipeline (e.g. `point_cloud/iteration_30000/point_cloud.ply`).

### Controls

- **Mouse drag** - orbit camera
- **Scroll** - zoom
- **Q** - quit

## How It Works

The rendering pipeline runs entirely on the GPU each frame:

1. **Preprocess** (compute) - Projects each 3D Gaussian to 2D screen space. Builds the 2D covariance from the 3D covariance via the Jacobian of the projective transform (`Sigma' = J W Sigma W^T J^T`). Computes the screen-space conic (inverse covariance), pixel radius (3-sigma), degree-0 SH color, and sigmoid opacity. Frustum culls and writes visible splats + depth sort keys.

2. **Sort** (compute) - Bitonic sort on depth keys to order splats back-to-front. Runs `O(log^2 N)` dispatches per frame with dynamic uniform offsets for sort parameters.

3. **Render** (vertex + fragment) - Draws instanced quads (6 vertices per splat) using `draw_indirect`. Each quad is expanded by the splat's pixel radius. The fragment shader evaluates the 2D Gaussian falloff (`exp(-0.5 * d^T * Sigma'^{-1} * d)`) and outputs premultiplied alpha. Hardware blending with `(One, OneMinusSrcAlpha)` composites back-to-front.

## Architecture

```
src/
  main.rs           - Entry point, State impl, pan-orbit camera, egui overlay
  gaussian.rs       - RawGaussian (PLY layout) / GpuGaussian (GPU-packed) structs
  ply.rs            - Binary PLY parser (bytemuck cast)
  splat_pass.rs     - PassNode<World> impl (buffers, pipelines, bind groups, dispatch)
  shaders/
    preprocess.wgsl - Compute: 3D->2D projection, covariance, cull, SH color
    sort.wgsl       - Compute: bitonic sort by depth
    render.wgsl     - Vertex+Fragment: instanced quads with Gaussian alpha blend
```

## Technical Details

- **SH degree-0 only** - View-independent color from the DC spherical harmonics coefficient
- **Bitonic sort** - Global GPU sort, no shared memory optimization; ~231 dispatches for 2M gaussians
- **No depth write** - Visibility handled entirely by sorted alpha blending
- **Premultiplied alpha** - Correct compositing via hardware blend state
- **PLY format** - Reads the standard 62-float-per-vertex binary layout (position, normals, SH DC, SH rest, opacity, scale, rotation)

## Prerequisites

- Rust (2024 edition)
- A GPU with WebGPU/Vulkan support
- A pre-trained 3DGS `.ply` file

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
