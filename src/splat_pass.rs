use crate::gaussian::GpuGaussian;
use nightshade::ecs::camera::queries::query_active_camera_matrices;
use nightshade::ecs::world::World;
use nightshade::prelude::wgpu;
use nightshade::render::wgpu::rendergraph::{PassExecutionContext, PassNode};
use wgpu::util::DeviceExt;

const PREPROCESS_SHADER: &str = include_str!("shaders/preprocess.wgsl");
const SORT_SHADER: &str = include_str!("shaders/sort.wgsl");
const RENDER_SHADER: &str = include_str!("shaders/render.wgsl");

const WORKGROUP_SIZE: u32 = 256;
const SORT_UNIFORM_ALIGNMENT: u64 = 256;
const SPLAT_SIZE: u64 = 48;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
    viewport: [f32; 2],
    focal: [f32; 2],
    gaussian_count: u32,
    padded_count: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SortUniforms {
    element_count: u32,
    block_size: u32,
    comparison_distance: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawIndirect {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

pub struct SplatPass {
    gaussian_count: u32,
    padded_count: u32,

    _gaussian_buffer: wgpu::Buffer,
    _splat_buffer: wgpu::Buffer,
    _sort_keys_buffer: wgpu::Buffer,
    _sort_values_buffer: wgpu::Buffer,
    draw_indirect_buffer: wgpu::Buffer,
    draw_indirect_reset_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,

    clear_sort_pipeline: wgpu::ComputePipeline,
    preprocess_pipeline: wgpu::ComputePipeline,
    sort_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,

    preprocess_bind_group: wgpu::BindGroup,
    sort_bind_group: wgpu::BindGroup,
    render_bind_group: wgpu::BindGroup,

    sort_stages: Vec<SortStage>,
}

struct SortStage {
    dynamic_offset: u32,
}

impl SplatPass {
    pub fn new(
        device: &wgpu::Device,
        gaussians: &[GpuGaussian],
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let gaussian_count = gaussians.len() as u32;
        let padded_count = gaussian_count.next_power_of_two();

        let gaussian_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gaussian Buffer"),
            contents: bytemuck::cast_slice(gaussians),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let splat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Splat Buffer"),
            size: SPLAT_SIZE * padded_count as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sort_keys_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sort Keys Buffer"),
            size: 4 * padded_count as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sort_values_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sort Values Buffer"),
            size: 4 * padded_count as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let draw_indirect_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Draw Indirect Buffer"),
            contents: bytemuck::cast_slice(&[DrawIndirect {
                vertex_count: 6,
                instance_count: 0,
                first_vertex: 0,
                first_instance: 0,
            }]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
        });

        let draw_indirect_reset_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Draw Indirect Reset Buffer"),
                contents: bytemuck::cast_slice(&[DrawIndirect {
                    vertex_count: 6,
                    instance_count: 0,
                    first_vertex: 0,
                    first_instance: 0,
                }]),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Splat Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sort_stages = compute_sort_stages(padded_count);
        let sort_uniform_data = build_sort_uniform_data(padded_count, &sort_stages);
        let sort_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sort Uniform Buffer"),
            contents: if sort_uniform_data.is_empty() {
                &[0u8; 256]
            } else {
                &sort_uniform_data
            },
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let preprocess_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Preprocess Bind Group Layout"),
                entries: &[
                    buffer_layout_entry(
                        0,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::BufferBindingType::Uniform,
                        false,
                    ),
                    buffer_layout_entry(
                        1,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        false,
                    ),
                    buffer_layout_entry(
                        2,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::BufferBindingType::Storage { read_only: false },
                        false,
                    ),
                    buffer_layout_entry(
                        3,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::BufferBindingType::Storage { read_only: false },
                        false,
                    ),
                    buffer_layout_entry(
                        4,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::BufferBindingType::Storage { read_only: false },
                        false,
                    ),
                    buffer_layout_entry(
                        5,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::BufferBindingType::Storage { read_only: false },
                        false,
                    ),
                ],
            });

        let sort_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Sort Bind Group Layout"),
                entries: &[
                    buffer_layout_entry(
                        0,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::BufferBindingType::Uniform,
                        true,
                    ),
                    buffer_layout_entry(
                        1,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::BufferBindingType::Storage { read_only: false },
                        false,
                    ),
                    buffer_layout_entry(
                        2,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::BufferBindingType::Storage { read_only: false },
                        false,
                    ),
                ],
            });

        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Render Bind Group Layout"),
                entries: &[
                    buffer_layout_entry(
                        0,
                        wgpu::ShaderStages::VERTEX,
                        wgpu::BufferBindingType::Uniform,
                        false,
                    ),
                    buffer_layout_entry(
                        1,
                        wgpu::ShaderStages::VERTEX,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        false,
                    ),
                    buffer_layout_entry(
                        2,
                        wgpu::ShaderStages::VERTEX,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        false,
                    ),
                ],
            });

        let preprocess_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Preprocess Bind Group"),
            layout: &preprocess_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gaussian_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: splat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sort_keys_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sort_values_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: draw_indirect_buffer.as_entire_binding(),
                },
            ],
        });

        let sort_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sort Bind Group"),
            layout: &sort_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &sort_uniform_buffer,
                        offset: 0,
                        size: Some(
                            std::num::NonZero::new(std::mem::size_of::<SortUniforms>() as u64)
                                .unwrap(),
                        ),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sort_keys_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sort_values_buffer.as_entire_binding(),
                },
            ],
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: splat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sort_values_buffer.as_entire_binding(),
                },
            ],
        });

        let preprocess_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Preprocess Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(PREPROCESS_SHADER)),
        });

        let sort_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sort Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SORT_SHADER)),
        });

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(RENDER_SHADER)),
        });

        let preprocess_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Preprocess Pipeline Layout"),
                bind_group_layouts: &[&preprocess_bind_group_layout],
                push_constant_ranges: &[],
            });

        let clear_sort_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Clear Sort Pipeline"),
                layout: Some(&preprocess_pipeline_layout),
                module: &preprocess_shader,
                entry_point: Some("clear_sort"),
                compilation_options: Default::default(),
                cache: None,
            });

        let preprocess_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Preprocess Pipeline"),
                layout: Some(&preprocess_pipeline_layout),
                module: &preprocess_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let sort_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sort Pipeline Layout"),
            bind_group_layouts: &[&sort_bind_group_layout],
            push_constant_ranges: &[],
        });

        let sort_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sort Pipeline"),
            layout: Some(&sort_pipeline_layout),
            module: &sort_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&render_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Splat Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vertex_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fragment_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            gaussian_count,
            padded_count,
            _gaussian_buffer: gaussian_buffer,
            _splat_buffer: splat_buffer,
            _sort_keys_buffer: sort_keys_buffer,
            _sort_values_buffer: sort_values_buffer,
            draw_indirect_buffer,
            draw_indirect_reset_buffer,
            uniform_buffer,
            clear_sort_pipeline,
            preprocess_pipeline,
            sort_pipeline,
            render_pipeline,
            preprocess_bind_group,
            sort_bind_group,
            render_bind_group,
            sort_stages,
        }
    }
}

impl PassNode<World> for SplatPass {
    fn name(&self) -> &str {
        "splat_pass"
    }

    fn reads(&self) -> Vec<&str> {
        vec![]
    }

    fn writes(&self) -> Vec<&str> {
        vec![]
    }

    fn reads_writes(&self) -> Vec<&str> {
        vec!["color", "depth"]
    }

    fn prepare(&mut self, _device: &wgpu::Device, queue: &wgpu::Queue, world: &World) {
        let camera_matrices = match query_active_camera_matrices(world) {
            Some(matrices) => matrices,
            None => return,
        };

        let view = camera_matrices.view;
        let projection = camera_matrices.projection;

        let (viewport_width, viewport_height) = world
            .resources
            .window
            .cached_viewport_size
            .unwrap_or((1920, 1080));

        let focal_x = projection[(0, 0)] * viewport_width as f32 * 0.5;
        let focal_y = projection[(1, 1)] * viewport_height as f32 * 0.5;

        let uniforms = Uniforms {
            view: view.into(),
            projection: projection.into(),
            viewport: [viewport_width as f32, viewport_height as f32],
            focal: [focal_x, focal_y],
            gaussian_count: self.gaussian_count,
            padded_count: self.padded_count,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }

    fn execute<'r, 'e>(
        &mut self,
        context: PassExecutionContext<'r, 'e, World>,
    ) -> nightshade::render::wgpu::rendergraph::Result<
        Vec<nightshade::render::wgpu::rendergraph::SubGraphRunCommand<'r>>,
    > {
        if self.gaussian_count == 0 {
            return Ok(context.into_sub_graph_commands());
        }

        context.encoder.copy_buffer_to_buffer(
            &self.draw_indirect_reset_buffer,
            0,
            &self.draw_indirect_buffer,
            0,
            std::mem::size_of::<DrawIndirect>() as u64,
        );

        {
            let mut compute_pass =
                context
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Clear Sort Pass"),
                        timestamp_writes: None,
                    });
            compute_pass.set_pipeline(&self.clear_sort_pipeline);
            compute_pass.set_bind_group(0, &self.preprocess_bind_group, &[]);
            let workgroups = self.padded_count.div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        {
            let mut compute_pass =
                context
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Preprocess Pass"),
                        timestamp_writes: None,
                    });
            compute_pass.set_pipeline(&self.preprocess_pipeline);
            compute_pass.set_bind_group(0, &self.preprocess_bind_group, &[]);
            let workgroups = self.gaussian_count.div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let num_pairs = self.padded_count / 2;
        let sort_workgroups = num_pairs.div_ceil(WORKGROUP_SIZE);

        for stage in &self.sort_stages {
            let mut compute_pass =
                context
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Sort Pass"),
                        timestamp_writes: None,
                    });
            compute_pass.set_pipeline(&self.sort_pipeline);
            compute_pass.set_bind_group(0, &self.sort_bind_group, &[stage.dynamic_offset]);
            compute_pass.dispatch_workgroups(sort_workgroups, 1, 1);
        }

        let (color_view, color_load, color_store) = context.get_color_attachment("color")?;
        let (depth_view, depth_load, depth_store) = context.get_depth_attachment("depth")?;

        {
            let mut render_pass = context
                .encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Splat Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: color_load,
                            store: color_store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: depth_load,
                            store: depth_store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.render_bind_group, &[]);
            render_pass.draw_indirect(&self.draw_indirect_buffer, 0);
        }

        Ok(context.into_sub_graph_commands())
    }
}

fn buffer_layout_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    buffer_type: wgpu::BufferBindingType,
    has_dynamic_offset: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: buffer_type,
            has_dynamic_offset,
            min_binding_size: None,
        },
        count: None,
    }
}

fn compute_sort_stages(padded_count: u32) -> Vec<SortStage> {
    if padded_count <= 1 {
        return vec![];
    }

    let mut stages = Vec::new();
    let mut stage_index = 0u32;

    let log_n = (padded_count as f64).log2() as u32;

    for outer_step in 0..log_n {
        let block_size = 2u32 << outer_step;
        let mut comparison_distance = block_size / 2;
        while comparison_distance >= 1 {
            stages.push(SortStage {
                dynamic_offset: stage_index * SORT_UNIFORM_ALIGNMENT as u32,
            });
            stage_index += 1;
            comparison_distance /= 2;
        }
    }

    stages
}

fn build_sort_uniform_data(padded_count: u32, stages: &[SortStage]) -> Vec<u8> {
    if stages.is_empty() {
        return vec![];
    }

    let log_n = (padded_count as f64).log2() as u32;
    let mut data = Vec::new();
    let mut stage_index = 0usize;

    for outer_step in 0..log_n {
        let block_size = 2u32 << outer_step;
        let mut comparison_distance = block_size / 2;
        while comparison_distance >= 1 {
            let offset_in_bytes = stage_index * SORT_UNIFORM_ALIGNMENT as usize;
            while data.len() < offset_in_bytes {
                data.push(0);
            }
            let params = SortUniforms {
                element_count: padded_count,
                block_size,
                comparison_distance,
                _pad: 0,
            };
            data.extend_from_slice(bytemuck::bytes_of(&params));
            stage_index += 1;
            comparison_distance /= 2;
        }
    }

    while !data.len().is_multiple_of(4) {
        data.push(0);
    }

    data
}
