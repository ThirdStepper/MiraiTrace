// Triangle rasterization with alpha blending (src-over compositing)
// Mirrors the CPU scanline rasterizer in raster.rs

struct Triangle {
    x0: i32, y0: i32,
    x1: i32, y1: i32,
    x2: i32, y2: i32,
    r: u32, g: u32, b: u32, a: u32,  // RGBA as separate u32 for alignment
}

struct Params {
    region_x: u32,
    region_y: u32,
    region_width: u32,
    region_height: u32,
    canvas_width: u32,
    canvas_height: u32,
    triangle_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(2) var<storage, read> triangle_indices: array<u32>;  // Indices into triangles array
@group(0) @binding(3) var<storage, read_write> output_buffer: array<atomic<u32>>;  // Region RGBA packed as u32

// Helper: Unpack u32 RGBA to vec4<f32>
fn unpack_rgba(packed: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(packed & 0xFFu),
        f32((packed >> 8u) & 0xFFu),
        f32((packed >> 16u) & 0xFFu),
        f32((packed >> 24u) & 0xFFu)
    );
}

// Helper: Pack vec4<f32> RGBA to u32
fn pack_rgba(color: vec4<f32>) -> u32 {
    let r = u32(clamp(color.r, 0.0, 255.0));
    let g = u32(clamp(color.g, 0.0, 255.0));
    let b = u32(clamp(color.b, 0.0, 255.0));
    let a = u32(clamp(color.a, 0.0, 255.0));
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

// Alpha blending: src-over in unpremultiplied space
// Mirrors blend_src_over_unpremul from raster.rs
fn blend_src_over(dst_packed: u32, src_rgba: vec4<u32>) -> u32 {
    let src_a = src_rgba.a;
    if (src_a == 0u) {
        return dst_packed;
    }

    let dst = unpack_rgba(dst_packed);
    let ia = 255u - src_a;

    // Integer math with rounding (matches CPU version)
    let r = (src_rgba.r * src_a + u32(dst.r) * ia + 127u) / 255u;
    let g = (src_rgba.g * src_a + u32(dst.g) * ia + 127u) / 255u;
    let b = (src_rgba.b * src_a + u32(dst.b) * ia + 127u) / 255u;
    let a_out = min(src_a + (u32(dst.a) * ia + 127u) / 255u, 255u);

    return pack_rgba(vec4<f32>(f32(r), f32(g), f32(b), f32(a_out)));
}

// Atomically blend a pixel using compare-and-swap loop
fn atomic_blend_pixel(pixel_idx: u32, src_rgba: vec4<u32>) {
    // Simple approach: use atomicLoad + atomicCompareExchangeWeak loop
    var old_val = atomicLoad(&output_buffer[pixel_idx]);

    loop {
        let new_val = blend_src_over(old_val, src_rgba);

        let exchange_result = atomicCompareExchangeWeak(&output_buffer[pixel_idx], old_val, new_val);
        if (exchange_result.exchanged) {
            break;
        }
        old_val = exchange_result.old_value;
    }
}

// Check if a point is inside a triangle using edge functions
fn point_in_triangle(px: i32, py: i32, tri: Triangle) -> bool {
    // Edge function: (p2.x - p1.x) * (p.y - p1.y) - (p2.y - p1.y) * (p.x - p1.x)
    let e0 = (tri.x1 - tri.x0) * (py - tri.y0) - (tri.y1 - tri.y0) * (px - tri.x0);
    let e1 = (tri.x2 - tri.x1) * (py - tri.y1) - (tri.y2 - tri.y1) * (px - tri.x1);
    let e2 = (tri.x0 - tri.x2) * (py - tri.y2) - (tri.y0 - tri.y2) * (px - tri.x2);

    // All edges must have same sign (or zero)
    return (e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0);
}

// Get triangle bounding box clamped to region
fn get_triangle_bbox(tri: Triangle) -> vec4<i32> {
    let min_x = min(min(tri.x0, tri.x1), tri.x2);
    let min_y = min(min(tri.y0, tri.y1), tri.y2);
    let max_x = max(max(tri.x0, tri.x1), tri.x2);
    let max_y = max(max(tri.y0, tri.y1), tri.y2);

    // Clamp to region bounds
    let region_x = i32(params.region_x);
    let region_y = i32(params.region_y);
    let region_x1 = region_x + i32(params.region_width) - 1;
    let region_y1 = region_y + i32(params.region_height) - 1;

    return vec4<i32>(
        clamp(min_x, region_x, region_x1),
        clamp(min_y, region_y, region_y1),
        clamp(max_x, region_x, region_x1),
        clamp(max_y, region_y, region_y1)
    );
}

// Main compute shader: each invocation processes one pixel
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    // Bounds check
    if (x >= params.region_width || y >= params.region_height) {
        return;
    }

    // Absolute canvas coordinates
    let canvas_x = i32(params.region_x + x);
    let canvas_y = i32(params.region_y + y);

    // Region buffer index
    let pixel_idx = y * params.region_width + x;

    // Process triangles in order (painter's algorithm)
    for (var i = 0u; i < params.triangle_count; i = i + 1u) {
        let tri_idx = triangle_indices[i];
        let tri = triangles[tri_idx];

        // Quick bbox rejection
        let bbox = get_triangle_bbox(tri);
        if (canvas_x < bbox.x || canvas_x > bbox.z || canvas_y < bbox.y || canvas_y > bbox.w) {
            continue;
        }

        // Point-in-triangle test
        if (point_in_triangle(canvas_x, canvas_y, tri)) {
            let src_rgba = vec4<u32>(tri.r, tri.g, tri.b, tri.a);
            atomic_blend_pixel(pixel_idx, src_rgba);
        }
    }
}
