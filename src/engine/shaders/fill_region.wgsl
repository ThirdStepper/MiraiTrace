// Fill a region buffer with a solid RGBA color
// This is a simple warmup shader for testing the GPU pipeline

struct Params {
    region_width: u32,
    region_height: u32,
    color_r: u32,
    color_g: u32,
    color_b: u32,
    color_a: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output_buffer: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    // Bounds check
    if (x >= params.region_width || y >= params.region_height) {
        return;
    }

    // Calculate pixel index
    let pixel_idx = y * params.region_width + x;

    // Pack RGBA into u32 (R=byte0, G=byte1, B=byte2, A=byte3)
    let color = params.color_r | (params.color_g << 8u) | (params.color_b << 16u) | (params.color_a << 24u);

    // Write to output buffer
    output_buffer[pixel_idx] = color;
}
