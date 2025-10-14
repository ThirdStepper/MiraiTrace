// Calculate Sum of Squared Errors (RGB only) between region buffer and target
// Uses parallel reduction with atomic accumulation

struct Params {
    region_x: u32,
    region_y: u32,
    region_width: u32,
    region_height: u32,
    canvas_width: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> region_buffer: array<u32>;  // Region RGBA packed as u32
@group(0) @binding(2) var<storage, read> target_buffer: array<u32>;  // Target RGBA packed as u32
@group(0) @binding(3) var<storage, read_write> output_sse: atomic<u32>;  // Accumulated SSE (will need two u32s for u64)

// Workgroup shared memory for local reduction
var<workgroup> shared_sse: array<u32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let x = global_id.x;
    let y = global_id.y;

    var local_sse: u32 = 0u;

    // Bounds check
    if (x < params.region_width && y < params.region_height) {
        // Region buffer index (linear in region)
        let region_idx = y * params.region_width + x;

        // Target buffer index (absolute canvas coordinates)
        let canvas_x = params.region_x + x;
        let canvas_y = params.region_y + y;
        let target_idx = canvas_y * params.canvas_width + canvas_x;

        // Read packed RGBA values
        let region_pixel = region_buffer[region_idx];
        let target_pixel = target_buffer[target_idx];

        // Unpack RGBA (R=byte0, G=byte1, B=byte2, A=byte3)
        let region_r = i32(region_pixel & 0xFFu);
        let region_g = i32((region_pixel >> 8u) & 0xFFu);
        let region_b = i32((region_pixel >> 16u) & 0xFFu);

        let target_r = i32(target_pixel & 0xFFu);
        let target_g = i32((target_pixel >> 8u) & 0xFFu);
        let target_b = i32((target_pixel >> 16u) & 0xFFu);

        // Calculate squared differences (RGB only, ignore alpha)
        let dr = region_r - target_r;
        let dg = region_g - target_g;
        let db = region_b - target_b;

        local_sse = u32(dr * dr + dg * dg + db * db);
    }

    // Store in shared memory
    shared_sse[local_idx] = local_sse;
    workgroupBarrier();

    // Parallel reduction within workgroup (256 threads -> 1 value)
    // Tree reduction: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    if (local_idx < 128u) {
        shared_sse[local_idx] = shared_sse[local_idx] + shared_sse[local_idx + 128u];
    }
    workgroupBarrier();

    if (local_idx < 64u) {
        shared_sse[local_idx] = shared_sse[local_idx] + shared_sse[local_idx + 64u];
    }
    workgroupBarrier();

    if (local_idx < 32u) {
        shared_sse[local_idx] = shared_sse[local_idx] + shared_sse[local_idx + 32u];
    }
    workgroupBarrier();

    if (local_idx < 16u) {
        shared_sse[local_idx] = shared_sse[local_idx] + shared_sse[local_idx + 16u];
    }
    workgroupBarrier();

    if (local_idx < 8u) {
        shared_sse[local_idx] = shared_sse[local_idx] + shared_sse[local_idx + 8u];
    }
    workgroupBarrier();

    if (local_idx < 4u) {
        shared_sse[local_idx] = shared_sse[local_idx] + shared_sse[local_idx + 4u];
    }
    workgroupBarrier();

    if (local_idx < 2u) {
        shared_sse[local_idx] = shared_sse[local_idx] + shared_sse[local_idx + 2u];
    }
    workgroupBarrier();

    if (local_idx == 0u) {
        let workgroup_sum = shared_sse[0] + shared_sse[1];
        atomicAdd(&output_sse, workgroup_sum);
    }
}
