// Batch proposal evaluation shader
// Evaluates multiple triangle proposals in parallel, one workgroup per proposal
// Returns SSE for each proposal

struct Triangle {
    x0: i32, y0: i32,
    x1: i32, y1: i32,
    x2: i32, y2: i32,
    r: u32, g: u32, b: u32, a: u32,
}

struct ProposalParams {
    region_x: u32,
    region_y: u32,
    region_width: u32,
    region_height: u32,
    triangle_start_idx: u32,  // Start index in triangles array
    triangle_count: u32,       // Number of triangles for this proposal
    bg_r: u32, bg_g: u32, bg_b: u32, bg_a: u32,
}

struct GlobalParams {
    canvas_width: u32,
    canvas_height: u32,
    proposal_count: u32,
}

@group(0) @binding(0) var<uniform> global_params: GlobalParams;
@group(0) @binding(1) var<storage, read> proposal_params: array<ProposalParams>;  // One per proposal
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;              // All triangles (shared)
@group(0) @binding(3) var<storage, read> triangle_indices: array<u32>;            // Draw order indices (shared)
@group(0) @binding(4) var<storage, read> target_buffer: array<u32>;               // Target image (packed RGBA)
@group(0) @binding(5) var<storage, read_write> output_sse: array<atomic<u32>>;    // SSE results (one per proposal)

// Shared memory for workgroup-level reduction
var<workgroup> shared_sse: array<u32, 256>;

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
fn blend_src_over(dst_packed: u32, src_rgba: vec4<u32>) -> u32 {
    let src_a = src_rgba.a;
    if (src_a == 0u) {
        return dst_packed;
    }

    let dst = unpack_rgba(dst_packed);
    let ia = 255u - src_a;

    let r = (src_rgba.r * src_a + u32(dst.r) * ia + 127u) / 255u;
    let g = (src_rgba.g * src_a + u32(dst.g) * ia + 127u) / 255u;
    let b = (src_rgba.b * src_a + u32(dst.b) * ia + 127u) / 255u;
    let a_out = min(src_a + (u32(dst.a) * ia + 127u) / 255u, 255u);

    return pack_rgba(vec4<f32>(f32(r), f32(g), f32(b), f32(a_out)));
}

// Check if a point is inside a triangle using edge functions
fn point_in_triangle(px: i32, py: i32, tri: Triangle) -> bool {
    let e0 = (tri.x1 - tri.x0) * (py - tri.y0) - (tri.y1 - tri.y0) * (px - tri.x0);
    let e1 = (tri.x2 - tri.x1) * (py - tri.y1) - (tri.y2 - tri.y1) * (px - tri.x1);
    let e2 = (tri.x0 - tri.x2) * (py - tri.y2) - (tri.y0 - tri.y2) * (px - tri.x2);

    return (e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0);
}

// Get triangle bounding box clamped to region
fn get_triangle_bbox(tri: Triangle, params: ProposalParams) -> vec4<i32> {
    let min_x = min(min(tri.x0, tri.x1), tri.x2);
    let min_y = min(min(tri.y0, tri.y1), tri.y2);
    let max_x = max(max(tri.x0, tri.x1), tri.x2);
    let max_y = max(max(tri.y0, tri.y1), tri.y2);

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

// Main compute shader: one workgroup per proposal, threads collaborate within workgroup
// Each thread processes multiple pixels
@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let proposal_id = workgroup_id.x;

    // Bounds check
    if (proposal_id >= global_params.proposal_count) {
        return;
    }

    let params = proposal_params[proposal_id];

    // Each thread processes one pixel in the region
    let x = local_id.x;
    let y = local_id.y;

    var local_sse: u32 = 0u;

    if (x < params.region_width && y < params.region_height) {
        // Absolute canvas coordinates
        let canvas_x = i32(params.region_x + x);
        let canvas_y = i32(params.region_y + y);

        // Start with background color
        let bg_color = pack_rgba(vec4<f32>(f32(params.bg_r), f32(params.bg_g), f32(params.bg_b), f32(params.bg_a)));
        var pixel_color = bg_color;

        // Rasterize all triangles for this proposal (in order)
        for (var i = 0u; i < params.triangle_count; i = i + 1u) {
            let idx_in_global = params.triangle_start_idx + i;
            let tri_idx = triangle_indices[idx_in_global];
            let tri = triangles[tri_idx];

            // Quick bbox rejection
            let bbox = get_triangle_bbox(tri, params);
            if (canvas_x < bbox.x || canvas_x > bbox.z || canvas_y < bbox.y || canvas_y > bbox.w) {
                continue;
            }

            // Point-in-triangle test
            if (point_in_triangle(canvas_x, canvas_y, tri)) {
                let src_rgba = vec4<u32>(tri.r, tri.g, tri.b, tri.a);
                pixel_color = blend_src_over(pixel_color, src_rgba);
            }
        }

        // Calculate SSE for this pixel
        let target_idx = u32(canvas_y) * global_params.canvas_width + u32(canvas_x);
        let target_pixel = target_buffer[target_idx];

        let rendered = unpack_rgba(pixel_color);
        let target = unpack_rgba(target_pixel);

        let dr = i32(rendered.r) - i32(target.r);
        let dg = i32(rendered.g) - i32(target.g);
        let db = i32(rendered.b) - i32(target.b);

        local_sse = u32(dr * dr + dg * dg + db * db);
    }

    // Store in shared memory
    shared_sse[local_idx] = local_sse;
    workgroupBarrier();

    // Parallel reduction within workgroup
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
        atomicAdd(&output_sse[proposal_id], workgroup_sum);
    }
}
