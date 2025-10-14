//! Compute backend abstraction for CPU vs GPU (WGPU)
//!
//! This module provides a trait-based interface for switching between different
//! compute backends for triangle rasterization and error calculation.

use super::{IntRect, Triangle};
use wgpu::util::DeviceExt;

/// Data for evaluating a single proposal on the GPU
#[derive(Clone, Debug)]
pub struct ProposalEvaluationData {
    /// The region affected by this proposal
    pub region: IntRect,
    /// All triangles in the candidate DNA
    pub triangles: Vec<Triangle>,
    /// Indices specifying draw order
    pub triangle_indices: Vec<usize>,
}

/// Compute backend trait - defines the interface for CPU/GPU implementations
#[allow(dead_code)]
pub trait ComputeBackend: Send + Sync {
    /// Initialize the backend (allocate buffers, compile shaders, etc.)
    fn initialize(&mut self, width: usize, height: usize) -> Result<(), String>;

    /// Rasterize triangles into a region buffer
    /// Returns the composited region as RGBA bytes
    fn compose_region(
        &mut self,
        region: &IntRect,
        canvas_w: usize,
        canvas_h: usize,
        background_color: [u8; 4],
        triangles: &[Triangle],
        triangle_indices: &[usize],
    ) -> Result<Vec<u8>, String>;

    /// Calculate squared error (RGB only) between two buffers in a region
    fn calculate_sse_region(
        &mut self,
        region_rgba: &[u8],
        target_rgba: &[u8],
        canvas_w: usize,
        region: &IntRect,
    ) -> Result<u64, String>;

    /// Evaluate multiple proposals in a single batch (GPU-optimized)
    /// Returns SSE for each proposal in the same order as input
    fn evaluate_proposals_batch(
        &mut self,
        canvas_rgba: &[u8],
        target_rgba: &[u8],
        canvas_w: usize,
        canvas_h: usize,
        background_color: [u8; 4],
        proposals: &[ProposalEvaluationData],
    ) -> Result<Vec<u64>, String>;

    /// Get backend name for display
    fn name(&self) -> &str;

    /// Check if backend is available
    fn is_available() -> bool where Self: Sized;

    /// Batch evaluation of multiple proposals - evaluates all proposals in parallel
    /// Returns SSE (sum of squared errors) for each proposal's region
    fn evaluate_proposals_batch(
        &mut self,
        canvas_rgba: &[u8],
        target_rgba: &[u8],
        canvas_w: usize,
        canvas_h: usize,
        background_color: [u8; 4],
        proposals: &[ProposalEvaluationData],
    ) -> Result<Vec<u64>, String>;
}

/// Data for a single proposal to be evaluated
#[derive(Clone, Debug)]
pub struct ProposalEvaluationData {
    pub region: IntRect,
    pub triangles: Vec<Triangle>,
    pub triangle_indices: Vec<usize>,
}

/// CPU backend - uses the existing CPU rasterization code
#[allow(dead_code)]
pub struct CpuBackend;

#[allow(dead_code)]
impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl ComputeBackend for CpuBackend {
    fn initialize(&mut self, _width: usize, _height: usize) -> Result<(), String> {
        Ok(())
    }

    fn compose_region(
        &mut self,
        region: &IntRect,
        canvas_w: usize,
        canvas_h: usize,
        background_color: [u8; 4],
        triangles: &[Triangle],
        triangle_indices: &[usize],
    ) -> Result<Vec<u8>, String> {
        use super::raster::{fill_region_rgba_solid, rasterize_triangle_over_unpremul_rgba_clipped_region};

        let mut buffer = vec![0u8; region.w * region.h * 4];
        fill_region_rgba_solid(&mut buffer, region, background_color);

        // Draw triangles in order
        for &idx in triangle_indices {
            if idx < triangles.len() {
                rasterize_triangle_over_unpremul_rgba_clipped_region(
                    &mut buffer,
                    region,
                    canvas_w,
                    canvas_h,
                    &triangles[idx],
                );
            }
        }

        Ok(buffer)
    }

    fn calculate_sse_region(
        &mut self,
        region_rgba: &[u8],
        target_rgba: &[u8],
        canvas_w: usize,
        region: &IntRect,
    ) -> Result<u64, String> {
        use super::raster::total_squared_error_rgb_region_from_buffer_vs_target;
        Ok(total_squared_error_rgb_region_from_buffer_vs_target(
            region_rgba, target_rgba, canvas_w, region
        ))
    }

    fn evaluate_proposals_batch(
        &mut self,
        _canvas_rgba: &[u8],
        target_rgba: &[u8],
        canvas_w: usize,
        canvas_h: usize,
        background_color: [u8; 4],
        proposals: &[ProposalEvaluationData],
    ) -> Result<Vec<u64>, String> {
        // CPU implementation: just loop through proposals sequentially
        let mut results = Vec::with_capacity(proposals.len());

        for proposal in proposals {
            // Compose the candidate region
            let region_buffer = self.compose_region(
                &proposal.region,
                canvas_w,
                canvas_h,
                background_color,
                &proposal.triangles,
                &proposal.triangle_indices,
            )?;

            // Calculate SSE
            let sse = self.calculate_sse_region(
                &region_buffer,
                target_rgba,
                canvas_w,
                &proposal.region,
            )?;

            results.push(sse);
        }

        Ok(results)
    }

    fn name(&self) -> &str {
        "CPU"
    }

    fn is_available() -> bool {
        true
    }

    fn evaluate_proposals_batch(
        &mut self,
        _canvas_rgba: &[u8],
        target_rgba: &[u8],
        canvas_w: usize,
        canvas_h: usize,
        background_color: [u8; 4],
        proposals: &[ProposalEvaluationData],
    ) -> Result<Vec<u64>, String> {
        use super::raster::{fill_region_rgba_solid, rasterize_triangle_over_unpremul_rgba_clipped_region, total_squared_error_rgb_region_from_buffer_vs_target};

        let mut results = Vec::with_capacity(proposals.len());

        for proposal in proposals {
            // Compose the region
            let mut buffer = vec![0u8; proposal.region.w * proposal.region.h * 4];
            fill_region_rgba_solid(&mut buffer, &proposal.region, background_color);

            // Draw triangles
            for &idx in &proposal.triangle_indices {
                if idx < proposal.triangles.len() {
                    rasterize_triangle_over_unpremul_rgba_clipped_region(
                        &mut buffer,
                        &proposal.region,
                        canvas_w,
                        canvas_h,
                        &proposal.triangles[idx],
                    );
                }
            }

            // Calculate SSE
            let sse = total_squared_error_rgb_region_from_buffer_vs_target(
                &buffer, target_rgba, canvas_w, &proposal.region
            );

            results.push(sse);
        }

        Ok(results)
    }
}

/// WGPU GPU backend with compiled shaders and pipelines
#[allow(dead_code)]
pub struct WgpuBackend {
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    width: usize,
    height: usize,

    // Shader pipelines
    fill_pipeline: Option<wgpu::ComputePipeline>,
    rasterize_pipeline: Option<wgpu::ComputePipeline>,
    sse_pipeline: Option<wgpu::ComputePipeline>,

    // Bind group layouts
    fill_bind_group_layout: Option<wgpu::BindGroupLayout>,
    rasterize_bind_group_layout: Option<wgpu::BindGroupLayout>,
    sse_bind_group_layout: Option<wgpu::BindGroupLayout>,

    // Persistent GPU buffers (reused across calls)
    target_buffer_gpu: Option<wgpu::Buffer>,          // Target image (packed RGBA u32) - rarely changes
    target_buffer_size: usize,                        // Current size of target buffer
    staging_buffer_read: Option<wgpu::Buffer>,        // Reusable staging buffer for GPU->CPU
    staging_buffer_size: usize,                       // Current size of staging buffer
    output_buffer: Option<wgpu::Buffer>,              // Reusable output buffer for compositing
    output_buffer_size: usize,                        // Current size of output buffer
}

#[allow(dead_code)]
impl WgpuBackend {
    pub fn new() -> Self {
        Self {
            device: None,
            queue: None,
            width: 0,
            height: 0,
            fill_pipeline: None,
            rasterize_pipeline: None,
            sse_pipeline: None,
            fill_bind_group_layout: None,
            rasterize_bind_group_layout: None,
            sse_bind_group_layout: None,
            target_buffer_gpu: None,
            target_buffer_size: 0,
            staging_buffer_read: None,
            staging_buffer_size: 0,
            output_buffer: None,
            output_buffer_size: 0,
        }
    }

    /// Check if WGPU is available (should be true on desktop with graphics drivers)
    pub fn check_available() -> bool {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }).await;

            adapter.is_some()
        })
    }

    /// Helper: Get or create staging buffer for reading from GPU
    fn get_or_create_staging_buffer(&mut self, size_bytes: usize) -> Result<&wgpu::Buffer, String> {
        let device = self.device.as_ref().ok_or("Device not initialized")?;

        // Reallocate if current buffer is too small
        if self.staging_buffer_size < size_bytes || self.staging_buffer_read.is_none() {
            let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Persistent Staging Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.staging_buffer_read = Some(new_buffer);
            self.staging_buffer_size = size_bytes;
        }

        Ok(self.staging_buffer_read.as_ref().unwrap())
    }

    /// Helper: Get or create output buffer for compositing
    fn get_or_create_output_buffer(&mut self, size_bytes: usize) -> Result<&wgpu::Buffer, String> {
        let device = self.device.as_ref().ok_or("Device not initialized")?;

        // Reallocate if current buffer is too small
        if self.output_buffer_size < size_bytes || self.output_buffer.is_none() {
            let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Persistent Output Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.output_buffer = Some(new_buffer);
            self.output_buffer_size = size_bytes;
        }

        Ok(self.output_buffer.as_ref().unwrap())
    }

    /// Helper: Upload target image to GPU (only if changed)
    fn upload_target_if_needed(&mut self, target_rgba: &[u8]) -> Result<&wgpu::Buffer, String> {
        let device = self.device.as_ref().ok_or("Device not initialized")?;

        let size_bytes = target_rgba.len();

        // Only re-upload if size changed or buffer doesn't exist
        // (In reality, target rarely changes, so this is a huge win)
        if self.target_buffer_size != size_bytes || self.target_buffer_gpu.is_none() {
            // Pack RGBA into u32
            let packed: Vec<u32> = target_rgba
                .chunks_exact(4)
                .map(|chunk| {
                    chunk[0] as u32
                        | ((chunk[1] as u32) << 8)
                        | ((chunk[2] as u32) << 16)
                        | ((chunk[3] as u32) << 24)
                })
                .collect();

            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Persistent Target Buffer"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

            self.target_buffer_gpu = Some(buffer);
            self.target_buffer_size = size_bytes;
        }

        Ok(self.target_buffer_gpu.as_ref().unwrap())
    }
}

impl ComputeBackend for WgpuBackend {
    fn initialize(&mut self, width: usize, height: usize) -> Result<(), String> {
        self.width = width;
        self.height = height;

        // Initialize WGPU asynchronously
        let (device, queue) = pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find GPU adapter")?;

            adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("MiraiTrace Compute Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create device: {}", e))
        })?;

        // Load and compile shaders
        let fill_shader_source = include_str!("shaders/fill_region.wgsl");
        let rasterize_shader_source = include_str!("shaders/rasterize_triangles.wgsl");
        let sse_shader_source = include_str!("shaders/calculate_sse.wgsl");

        let fill_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fill Region Shader"),
            source: wgpu::ShaderSource::Wgsl(fill_shader_source.into()),
        });

        let rasterize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Rasterize Triangles Shader"),
            source: wgpu::ShaderSource::Wgsl(rasterize_shader_source.into()),
        });

        let sse_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Calculate SSE Shader"),
            source: wgpu::ShaderSource::Wgsl(sse_shader_source.into()),
        });

        // Create bind group layouts
        let fill_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fill Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let rasterize_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Rasterize Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let sse_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSE Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create compute pipelines
        let fill_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fill Pipeline Layout"),
            bind_group_layouts: &[&fill_bind_group_layout],
            push_constant_ranges: &[],
        });

        let fill_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fill Pipeline"),
            layout: Some(&fill_pipeline_layout),
            module: &fill_shader,
            entry_point: "main",
        });

        let rasterize_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Rasterize Pipeline Layout"),
            bind_group_layouts: &[&rasterize_bind_group_layout],
            push_constant_ranges: &[],
        });

        let rasterize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Rasterize Pipeline"),
            layout: Some(&rasterize_pipeline_layout),
            module: &rasterize_shader,
            entry_point: "main",
        });

        let sse_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSE Pipeline Layout"),
            bind_group_layouts: &[&sse_bind_group_layout],
            push_constant_ranges: &[],
        });

        let sse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SSE Pipeline"),
            layout: Some(&sse_pipeline_layout),
            module: &sse_shader,
            entry_point: "main",
        });

        // Store everything
        self.device = Some(device);
        self.queue = Some(queue);
        self.fill_pipeline = Some(fill_pipeline);
        self.rasterize_pipeline = Some(rasterize_pipeline);
        self.sse_pipeline = Some(sse_pipeline);
        self.fill_bind_group_layout = Some(fill_bind_group_layout);
        self.rasterize_bind_group_layout = Some(rasterize_bind_group_layout);
        self.sse_bind_group_layout = Some(sse_bind_group_layout);

        Ok(())
    }

    fn compose_region(
        &mut self,
        region: &IntRect,
        canvas_w: usize,
        canvas_h: usize,
        background_color: [u8; 4],
        triangles: &[Triangle],
        triangle_indices: &[usize],
    ) -> Result<Vec<u8>, String> {
        if region.w == 0 || region.h == 0 {
            return Ok(Vec::new());
        }

        let pixel_count = region.w * region.h;
        let buffer_size = pixel_count * 4;

        // Prepare persistent buffers (borrows self mutably, then releases)
        self.get_or_create_output_buffer(buffer_size)?;
        self.get_or_create_staging_buffer(buffer_size)?;

        // Now we can borrow device/queue/pipelines immutably without conflicts
        let device = self.device.as_ref().ok_or("Device not initialized")?;
        let queue = self.queue.as_ref().ok_or("Queue not initialized")?;
        let fill_pipeline = self.fill_pipeline.as_ref().ok_or("Fill pipeline not initialized")?;
        let rasterize_pipeline = self.rasterize_pipeline.as_ref().ok_or("Rasterize pipeline not initialized")?;
        let fill_bgl = self.fill_bind_group_layout.as_ref().ok_or("Fill bind group layout not initialized")?;
        let rasterize_bgl = self.rasterize_bind_group_layout.as_ref().ok_or("Rasterize bind group layout not initialized")?;
        let output_buffer = self.output_buffer.as_ref().ok_or("Output buffer not created")?;
        let staging_buffer = self.staging_buffer_read.as_ref().ok_or("Staging buffer not created")?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compose Region Encoder"),
        });

        // Step 1: Fill with background color
        {
            let params_data: [u32; 8] = [
                region.w as u32,
                region.h as u32,
                background_color[0] as u32,
                background_color[1] as u32,
                background_color[2] as u32,
                background_color[3] as u32,
                0, 0, // padding
            ];

            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Fill Params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Fill Bind Group"),
                layout: fill_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fill Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(fill_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_x = (region.w as u32 + 15) / 16;
            let workgroup_y = (region.h as u32 + 15) / 16;
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        // Step 2: Rasterize triangles
        if !triangle_indices.is_empty() {
            // Prepare triangle data (convert to GPU format)
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct GpuTriangle {
                x0: i32, y0: i32,
                x1: i32, y1: i32,
                x2: i32, y2: i32,
                r: u32, g: u32, b: u32, a: u32,
                _pad: [u32; 2], // Padding for alignment
            }

            let gpu_triangles: Vec<GpuTriangle> = triangles.iter().map(|t| GpuTriangle {
                x0: t.x0, y0: t.y0,
                x1: t.x1, y1: t.y1,
                x2: t.x2, y2: t.y2,
                r: t.r as u32,
                g: t.g as u32,
                b: t.b as u32,
                a: t.a as u32,
                _pad: [0, 0],
            }).collect();

            let triangle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Triangle Buffer"),
                contents: bytemuck::cast_slice(&gpu_triangles),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let indices_u32: Vec<u32> = triangle_indices.iter().map(|&i| i as u32).collect();
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&indices_u32),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let params_data: [u32; 8] = [
                region.x as u32,
                region.y as u32,
                region.w as u32,
                region.h as u32,
                canvas_w as u32,
                canvas_h as u32,
                triangle_indices.len() as u32,
                0, // padding
            ];

            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Rasterize Params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Rasterize Bind Group"),
                layout: rasterize_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: triangle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: index_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Rasterize Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(rasterize_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_x = (region.w as u32 + 15) / 16;
            let workgroup_y = (region.h as u32 + 15) / 16;
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        // Copy results to staging buffer for readback
        encoder.copy_buffer_to_buffer(output_buffer, 0, staging_buffer, 0, buffer_size as u64);

        queue.submit(Some(encoder.finish()));

        // Map and read buffer
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().map_err(|e| format!("Buffer map failed: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    fn calculate_sse_region(
        &mut self,
        region_rgba: &[u8],
        target_rgba: &[u8],
        canvas_w: usize,
        region: &IntRect,
    ) -> Result<u64, String> {
        if region.w == 0 || region.h == 0 {
            return Ok(0);
        }

        // Upload target buffer if needed (borrows self mutably, then releases)
        self.upload_target_if_needed(target_rgba)?;

        // Now borrow immutably
        let device = self.device.as_ref().ok_or("Device not initialized")?;
        let queue = self.queue.as_ref().ok_or("Queue not initialized")?;
        let sse_pipeline = self.sse_pipeline.as_ref().ok_or("SSE pipeline not initialized")?;
        let sse_bgl = self.sse_bind_group_layout.as_ref().ok_or("SSE bind group layout not initialized")?;
        let target_buffer = self.target_buffer_gpu.as_ref().ok_or("Target buffer not created")?;

        // Pack region RGBA bytes into u32 for GPU (R|G<<8|B<<16|A<<24)
        let region_packed: Vec<u32> = region_rgba
            .chunks_exact(4)
            .map(|chunk| {
                chunk[0] as u32
                    | ((chunk[1] as u32) << 8)
                    | ((chunk[2] as u32) << 16)
                    | ((chunk[3] as u32) << 24)
            })
            .collect();

        // Create region buffer (changes every call)
        let region_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Region Buffer"),
            contents: bytemuck::cast_slice(&region_packed),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Output buffer for SSE accumulation (single atomic u32, we'll handle u64 in CPU)
        let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SSE Output"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Params
        let params_data: [u32; 8] = [
            region.x as u32,
            region.y as u32,
            region.w as u32,
            region.h as u32,
            canvas_w as u32,
            0, 0, 0, // padding
        ];

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SSE Params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSE Bind Group"),
            layout: sse_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: region_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: target_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SSE Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SSE Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(sse_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_x = (region.w as u32 + 15) / 16;
            let workgroup_y = (region.h as u32 + 15) / 16;
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        // Read back result
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SSE Staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, 4);
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().map_err(|e| format!("Buffer map failed: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let sse_u32 = bytemuck::cast_slice::<u8, u32>(&data)[0];
        drop(data);
        staging_buffer.unmap();

        Ok(sse_u32 as u64)
    }

    fn evaluate_proposals_batch(
        &mut self,
        _canvas_rgba: &[u8],
        target_rgba: &[u8],
        canvas_w: usize,
        canvas_h: usize,
        background_color: [u8; 4],
        proposals: &[ProposalEvaluationData],
    ) -> Result<Vec<u64>, String> {
        if proposals.is_empty() {
            return Ok(Vec::new());
        }

        // For GPU batch evaluation, we have two strategies:
        // 1. Simple: Sequential evaluation (what we do now)
        // 2. Advanced: True parallel batch with custom shader (future optimization)

        // Currently using strategy 1 (still benefits from persistent buffers)
        // Strategy 2 would require:
        // - Packing all proposal data into unified buffers
        // - Dispatching one workgroup per proposal
        // - Each workgroup rasterizes + calculates SSE independently
        // - Results written to output array

        let mut results = Vec::with_capacity(proposals.len());

        for proposal in proposals {
            // Compose the candidate region
            let region_buffer = self.compose_region(
                &proposal.region,
                canvas_w,
                canvas_h,
                background_color,
                &proposal.triangles,
                &proposal.triangle_indices,
            )?;

            // Calculate SSE
            let sse = self.calculate_sse_region(
                &region_buffer,
                target_rgba,
                canvas_w,
                &proposal.region,
            )?;

            results.push(sse);
        }

        Ok(results)
    }

    fn name(&self) -> &str {
        "WGPU GPU"
    }

    fn is_available() -> bool {
        WgpuBackend::check_available()
    }

    fn evaluate_proposals_batch(
        &mut self,
        _canvas_rgba: &[u8],
        target_rgba: &[u8],
        canvas_w: usize,
        canvas_h: usize,
        background_color: [u8; 4],
        proposals: &[ProposalEvaluationData],
    ) -> Result<Vec<u64>, String> {
        // For simplicity, use sequential CPU evaluation for each proposal
        // Full GPU batch evaluation would require a new shader that processes multiple regions
        // This still uses GPU acceleration for each individual proposal
        let mut results = Vec::with_capacity(proposals.len());

        for proposal in proposals {
            // Use GPU to compose the region
            let buffer = self.compose_region(
                &proposal.region,
                canvas_w,
                canvas_h,
                background_color,
                &proposal.triangles,
                &proposal.triangle_indices,
            )?;

            // Use GPU to calculate SSE
            let sse = self.calculate_sse_region(
                &buffer,
                target_rgba,
                canvas_w,
                &proposal.region,
            )?;

            results.push(sse);
        }

        Ok(results)
    }
}
