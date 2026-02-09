struct GpuGaussian {
    position: vec3<f32>,
    opacity_logit: f32,
    sh_dc: vec3<f32>,
    _pad0: f32,
    scale_log: vec3<f32>,
    _pad1: f32,
    rotation: vec4<f32>,
};

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

struct DrawIndirect {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<GpuGaussian>;
@group(0) @binding(2) var<storage, read_write> splats: array<Splat2D>;
@group(0) @binding(3) var<storage, read_write> sort_keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> sort_values: array<u32>;
@group(0) @binding(5) var<storage, read_write> draw_indirect: DrawIndirect;

@compute @workgroup_size(256)
fn clear_sort(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= uniforms.padded_count {
        return;
    }
    sort_keys[index] = 0xFFFFFFFFu;
    sort_values[index] = index % uniforms.gaussian_count;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= uniforms.gaussian_count {
        return;
    }

    let gaussian = gaussians[index];

    let view_pos = uniforms.view * vec4<f32>(gaussian.position, 1.0);

    if view_pos.z >= -0.1 {
        return;
    }

    let clip_pos = uniforms.projection * view_pos;
    let ndc = clip_pos.xyz / clip_pos.w;

    if abs(ndc.x) > 1.3 || abs(ndc.y) > 1.3 {
        return;
    }

    let screen_x = (ndc.x * 0.5 + 0.5) * uniforms.viewport.x;
    let screen_y = (ndc.y * -0.5 + 0.5) * uniforms.viewport.y;

    let quat = gaussian.rotation;
    let r = quat.x;
    let x = quat.y;
    let y = quat.z;
    let z = quat.w;

    let rotation_matrix = mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + r * z), 2.0 * (x * z - r * y)),
        vec3<f32>(2.0 * (x * y - r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + r * x)),
        vec3<f32>(2.0 * (x * z + r * y), 2.0 * (y * z - r * x), 1.0 - 2.0 * (x * x + y * y)),
    );

    let scale = exp(gaussian.scale_log);
    let scale_matrix = mat3x3<f32>(
        vec3<f32>(scale.x, 0.0, 0.0),
        vec3<f32>(0.0, scale.y, 0.0),
        vec3<f32>(0.0, 0.0, scale.z),
    );

    let m = rotation_matrix * scale_matrix;
    let sigma = m * transpose(m);

    let view3x3 = mat3x3<f32>(
        uniforms.view[0].xyz,
        uniforms.view[1].xyz,
        uniforms.view[2].xyz,
    );

    let tz = view_pos.z;
    let focal_x = uniforms.focal.x;
    let focal_y = uniforms.focal.y;

    let jacobian = mat3x3<f32>(
        vec3<f32>(focal_x / tz, 0.0, 0.0),
        vec3<f32>(0.0, focal_y / tz, 0.0),
        vec3<f32>(-focal_x * view_pos.x / (tz * tz), -focal_y * view_pos.y / (tz * tz), 0.0),
    );

    let t = jacobian * view3x3;
    let cov2d = t * sigma * transpose(t);

    let cov_a = cov2d[0][0] + 0.3;
    let cov_b = cov2d[0][1];
    let cov_d = cov2d[1][1] + 0.3;

    let det = cov_a * cov_d - cov_b * cov_b;
    if det <= 0.0 {
        return;
    }

    let det_inv = 1.0 / det;
    let conic = vec3<f32>(cov_d * det_inv, -cov_b * det_inv, cov_a * det_inv);

    let mid = 0.5 * (cov_a + cov_d);
    let discriminant = max(mid * mid - det, 0.0);
    let lambda_max = mid + sqrt(discriminant);
    let pixel_radius = ceil(3.0 * sqrt(lambda_max));

    if pixel_radius <= 0.0 || pixel_radius > 1024.0 {
        return;
    }

    let opacity = 1.0 / (1.0 + exp(-gaussian.opacity_logit));

    if opacity < 1.0 / 255.0 {
        return;
    }

    let sh_c0 = 0.2820947917738781;
    let color = vec3<f32>(
        gaussian.sh_dc.x * sh_c0 + 0.5,
        gaussian.sh_dc.y * sh_c0 + 0.5,
        gaussian.sh_dc.z * sh_c0 + 0.5,
    );
    let clamped_color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));

    let slot = atomicAdd(&draw_indirect.instance_count, 1u);

    let depth = -view_pos.z;
    let depth_key = 0xFFFFFFFFu - bitcast<u32>(depth);

    sort_keys[slot] = depth_key;
    sort_values[slot] = slot;

    splats[slot] = Splat2D(
        vec4<f32>(clamped_color, 1.0),
        vec4<f32>(conic, opacity),
        vec2<f32>(screen_x, screen_y),
        pixel_radius,
        0.0,
    );
}
