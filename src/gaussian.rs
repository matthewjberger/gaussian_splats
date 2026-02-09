#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RawGaussian {
    pub position: [f32; 3],
    pub normals: [f32; 3],
    pub sh_dc: [f32; 3],
    pub sh_rest: [f32; 45],
    pub opacity: f32,
    pub scale: [f32; 3],
    pub rotation: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuGaussian {
    pub position: [f32; 3],
    pub opacity_logit: f32,
    pub sh_dc: [f32; 3],
    pub _pad0: f32,
    pub scale_log: [f32; 3],
    pub _pad1: f32,
    pub rotation: [f32; 4],
}

impl From<&RawGaussian> for GpuGaussian {
    fn from(raw: &RawGaussian) -> Self {
        let [qw, qx, qy, qz] = raw.rotation;
        let length = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt().max(1e-8);
        Self {
            position: raw.position,
            opacity_logit: raw.opacity,
            sh_dc: raw.sh_dc,
            _pad0: 0.0,
            scale_log: raw.scale,
            _pad1: 0.0,
            rotation: [qw / length, qx / length, qy / length, qz / length],
        }
    }
}
