struct Splat2D {
    color: vec4<f32>,
    conic_and_opacity: vec4<f32>,
    center: vec2<f32>,
    radius: f32,
    _pad: f32,
};

struct Uniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>,
    gaussian_count: u32,
    padded_count: u32,
    _pad1: u32,
    _pad2: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) splat_color: vec4<f32>,
    @location(1) conic_and_opacity: vec4<f32>,
    @location(2) offset: vec2<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> splats: array<Splat2D>;
@group(0) @binding(2) var<storage, read> sort_values: array<u32>;

const QUAD_VERTICES: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
);

@vertex
fn vertex_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let sorted_index = sort_values[instance_index];
    let splat = splats[sorted_index];

    let quad_offset = QUAD_VERTICES[vertex_index % 6u];
    let pixel_offset = quad_offset * splat.radius;

    let screen_pos = splat.center + pixel_offset;

    let ndc_x = screen_pos.x / uniforms.viewport.x * 2.0 - 1.0;
    let ndc_y = screen_pos.y / uniforms.viewport.y * 2.0 - 1.0;

    var output: VertexOutput;
    output.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    output.splat_color = splat.color;
    output.conic_and_opacity = splat.conic_and_opacity;
    output.offset = pixel_offset;
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let offset = input.offset;
    let conic = input.conic_and_opacity.xyz;
    let opacity = input.conic_and_opacity.w;

    let power = -0.5 * (conic.x * offset.x * offset.x + 2.0 * conic.y * offset.x * offset.y + conic.z * offset.y * offset.y);

    if power > 0.0 {
        discard;
    }

    let alpha = min(0.99, opacity * exp(power));

    if alpha < 1.0 / 255.0 {
        discard;
    }

    return vec4<f32>(input.splat_color.rgb * alpha, alpha);
}
