// -----------------------------------------------------------------------------
// Rasterization & blending
// -----------------------------------------------------------------------------
use super::{IntRect, Triangle, TriangleDna};
use super::TriangleSpatialIndex;


#[inline]
pub(crate) fn blend_src_over_unpremul(dst: &mut [u8], src: &[u8]) {
    // src-over in unpremultiplied space, integer math with rounding.
    let a = src[3] as u32;
    if a == 0 { return; }
    let ia = 255 - a;

    let r = ((src[0] as u32 * a) + (dst[0] as u32 * ia) + 127) / 255;
    let g = ((src[1] as u32 * a) + (dst[1] as u32 * ia) + 127) / 255;
    let b = ((src[2] as u32 * a) + (dst[2] as u32 * ia) + 127) / 255;
    let da = dst[3] as u32;
    let a_out = (a + (da * ia + 127) / 255).min(255);

    dst[0] = r as u8;
    dst[1] = g as u8;
    dst[2] = b as u8;
    dst[3] = a_out as u8;
}

/// Fill a region buffer with a solid background color.
pub(crate) fn fill_region_rgba_solid(out_rgba_region: &mut [u8], region: &IntRect, color: [u8; 4]) {
    if region.w == 0 || region.h == 0 {
        return;
    }
    for y in 0..region.h {
        for x in 0..region.w {
            let i = (y * region.w + x) * 4;
            out_rgba_region[i..i + 4].copy_from_slice(&color);
        }
    }
}

/// Fills a triangle using scanlines into a *region buffer* (unpremul RGBA),
/// clipped to `clip_region` (the region rectangle in canvas coords).
#[inline]
pub(crate) fn rasterize_triangle_over_unpremul_rgba_clipped_region(
    out_region_rgba: &mut [u8],
    clip_region: &IntRect,   // x,y,w,h are usize
    canvas_w: usize,
    canvas_h: usize,
    tri: &Triangle,
) {
    if clip_region.w == 0 || clip_region.h == 0 { return; }

    // Triangle vertices (i32)
    let (mut x0, mut y0) = (tri.x0, tri.y0);
    let (mut x1, mut y1) = (tri.x1, tri.y1);
    let (mut x2, mut y2) = (tri.x2, tri.y2);

    // Sort by y (y0 <= y1 <= y2)
    if y1 < y0 { core::mem::swap(&mut y0, &mut y1); core::mem::swap(&mut x0, &mut x1); }
    if y2 < y0 { core::mem::swap(&mut y0, &mut y2); core::mem::swap(&mut x0, &mut x2); }
    if y2 < y1 { core::mem::swap(&mut y1, &mut y2); core::mem::swap(&mut x1, &mut x2); }
    if y0 == y2 { return; } // degenerate

    // ---- clip rect to i32 once ----
    let clip_x  = clip_region.x as i32;
    let clip_y  = clip_region.y as i32;
    let clip_w  = clip_region.w as i32;
    let clip_h  = clip_region.h as i32;
    let clip_x1 = (clip_x + clip_w - 1).min(canvas_w as i32 - 1);
    let clip_y1 = (clip_y + clip_h - 1).min(canvas_h as i32 - 1);

    // Vertical clip
    let y_min = y0.max(clip_y).max(0);
    let y_max = y2.min(clip_y1).min(canvas_h as i32 - 1);
    if y_min > y_max { return; }

    #[inline]
    fn edge_step_fixed(x0: i32, y0: i32, x1: i32, y1: i32) -> (i64, i64, i32, i32) {
        if y0 == y1 {
            return (((x0 as i64) << 16), 0i64, y0, y0);
        }
        let (y_min, y_max) = (y0.min(y1), y0.max(y1));
        let x_at_ymin_fp: i64 = (if y0 < y1 { x0 } else { x1 } as i64) << 16; // 16.16
        let dx_dy_fp: i64 = (((x1 as i64) - (x0 as i64)) << 16) / ((y1 - y0) as i64);
        (x_at_ymin_fp, dx_dy_fp, y_min, y_max)
    }

    // Edges in fixed-point
    let (mut x01_fp, dx01_fp, y01_min, _) = edge_step_fixed(x0, y0, x1, y1);
    let (mut x02_fp, dx02_fp, y02_min, _) = edge_step_fixed(x0, y0, x2, y2);
    let (mut x12_fp, dx12_fp, y12_min, _) = edge_step_fixed(x1, y1, x2, y2);

    // Advance x01/x02 to y_min (i64 math to avoid overflow)
    if y_min > y01_min { x01_fp += dx01_fp * ((y_min - y01_min) as i64); }
    if y_min > y02_min { x02_fp += dx02_fp * ((y_min - y02_min) as i64); }

    let src_rgba = [tri.r, tri.g, tri.b, tri.a];
    let region_w = clip_region.w as usize;

    #[inline]
    fn blend_span_unpremul_rgba_region(
        out_region_rgba: &mut [u8],
        region_w: usize,
        row_in_region: usize,
        x0_in_region: usize,
        x1_in_region_excl: usize,
        src_rgba: &[u8; 4],
    ) {
        if x0_in_region >= x1_in_region_excl { return; }
        let row = &mut out_region_rgba[row_in_region * region_w * 4 .. (row_in_region + 1) * region_w * 4];
        let mut i = x0_in_region * 4;
        let end = x1_in_region_excl * 4;
        while i < end {
            blend_src_over_unpremul(&mut row[i .. i + 4], src_rgba);
            i += 4;
        }
    }

    // Upper half: [y0, y1)
    #[allow(non_snake_case)]
    if y_min < y1 && y_min <= y_max {
        let mut y = y_min;
        while y < y1 && y <= y_max {
            let xL_fp = x01_fp.min(x02_fp);
            let xR_fp = x01_fp.max(x02_fp);
            let xl = ((xL_fp >> 16) as i32).max(clip_x).max(0);
            let xr = ((xR_fp >> 16) as i32).min(clip_x1).min(canvas_w as i32 - 1);


            if xl <= xr {
                let ry  = (y  - clip_y) as usize;
                let rx0 = (xl - clip_x) as usize;
                let rx1 = (xr - clip_x + 1) as usize; // exclusive
                blend_span_unpremul_rgba_region(out_region_rgba, region_w, ry, rx0, rx1, &src_rgba);
            }

            x01_fp += dx01_fp;
            x02_fp += dx02_fp;
            y += 1;
        }
    }

    // Lower half: [y1, y2]
    #[allow(non_snake_case)]
    if y_max >= y1 {
        let start_y = y_min.max(y1);

        // Recompute x12_fp and x02_fp at start_y
        if start_y > y12_min {
            let base_x12_fp = (if y1 < y2 { x1 } else { x2 } as i64) << 16;
            x12_fp = base_x12_fp + dx12_fp * ((start_y - y12_min) as i64);
        }
        if start_y > y02_min {
            let base_x02_fp = (if y0 < y2 { x0 } else { x2 } as i64) << 16;
            x02_fp = base_x02_fp + dx02_fp * ((start_y - y02_min) as i64);
        }

        let mut y = start_y;
        while y <= y_max {
            let xL_fp = x12_fp.min(x02_fp);
            let xR_fp = x12_fp.max(x02_fp);
            let xl = ((xL_fp >> 16) as i32).max(clip_x).max(0);
            let xr = ((xR_fp >> 16) as i32).min(clip_x1).min(canvas_w as i32 - 1);

            if xl <= xr {
                let ry  = (y  - clip_y) as usize;
                let rx0 = (xl - clip_x) as usize;
                let rx1 = (xr - clip_x + 1) as usize;
                blend_span_unpremul_rgba_region(out_region_rgba, region_w, ry, rx0, rx1, &src_rgba);
            }

            x12_fp += dx12_fp;
            x02_fp += dx02_fp;
            y += 1;
        }
    }
}

/// Compute SSE (RGB only) between `region_rgba` and the target image for that region.
pub(crate) fn total_squared_error_rgb_region_from_buffer_vs_target(
    region_rgba: &[u8],
    target_rgba: &[u8],
    canvas_w: usize,
    region: &IntRect,
) -> u64 {
    let mut sse: u64 = 0;
    for y in 0..region.h {
        let base_canvas = ((region.y + y) * canvas_w + region.x) * 4;
        let base_region = (y * region.w) * 4;
        let mut i = 0usize;
        let end = region.w * 4;

        while i + 16 <= end {
            // 4 pixels
            let a = base_region + i;
            let b = base_canvas + i;

            let dr0 = region_rgba[a] as i32 - target_rgba[b] as i32;
            let dg0 = region_rgba[a + 1] as i32 - target_rgba[b + 1] as i32;
            let db0 = region_rgba[a + 2] as i32 - target_rgba[b + 2] as i32;

            let dr1 = region_rgba[a + 4] as i32 - target_rgba[b + 4] as i32;
            let dg1 = region_rgba[a + 5] as i32 - target_rgba[b + 5] as i32;
            let db1 = region_rgba[a + 6] as i32 - target_rgba[b + 6] as i32;

            let dr2 = region_rgba[a + 8] as i32 - target_rgba[b + 8] as i32;
            let dg2 = region_rgba[a + 9] as i32 - target_rgba[b + 9] as i32;
            let db2 = region_rgba[a +10] as i32 - target_rgba[b +10] as i32;

            let dr3 = region_rgba[a +12] as i32 - target_rgba[b +12] as i32;
            let dg3 = region_rgba[a +13] as i32 - target_rgba[b +13] as i32;
            let db3 = region_rgba[a +14] as i32 - target_rgba[b +14] as i32;

            sse = sse.wrapping_add((dr0*dr0 + dg0*dg0 + db0*db0) as u64);
            sse = sse.wrapping_add((dr1*dr1 + dg1*dg1 + db1*db1) as u64);
            sse = sse.wrapping_add((dr2*dr2 + dg2*dg2 + db2*db2) as u64);
            sse = sse.wrapping_add((dr3*dr3 + dg3*dg3 + db3*db3) as u64);

            i += 16;
        }

        // tail
        while i < end {
            let a = base_region + i;
            let b = base_canvas + i;
            let dr = region_rgba[a] as i32 - target_rgba[b] as i32;
            let dg = region_rgba[a + 1] as i32 - target_rgba[b + 1] as i32;
            let db = region_rgba[a + 2] as i32 - target_rgba[b + 2] as i32;
            sse = sse.wrapping_add((dr*dr + dg*dg + db*db) as u64);
            i += 4;
        }
    }
    sse
}

#[inline]
pub(crate) fn total_squared_error_rgb_region_from_canvas_vs_target(
    canvas_rgba: &[u8],
    target_rgba: &[u8],
    canvas_w: usize,
    region: &IntRect,
) -> u64 {
    let mut sse: u64 = 0;
    for y in 0..region.h {
        // both sides use the same base (full-canvas offset)
        let base = ((region.y + y) * canvas_w + region.x) * 4;
        let mut i = 0usize;
        let end = region.w * 4;

        // 4px unrolled
        while i + 16 <= end {
            let a = base + i;
            let b = base + i;

            let dr0 = canvas_rgba[a] as i32 - target_rgba[b] as i32;
            let dg0 = canvas_rgba[a + 1] as i32 - target_rgba[b + 1] as i32;
            let db0 = canvas_rgba[a + 2] as i32 - target_rgba[b + 2] as i32;

            let dr1 = canvas_rgba[a + 4] as i32 - target_rgba[b + 4] as i32;
            let dg1 = canvas_rgba[a + 5] as i32 - target_rgba[b + 5] as i32;
            let db1 = canvas_rgba[a + 6] as i32 - target_rgba[b + 6] as i32;

            let dr2 = canvas_rgba[a + 8] as i32 - target_rgba[b + 8] as i32;
            let dg2 = canvas_rgba[a + 9] as i32 - target_rgba[b + 9] as i32;
            let db2 = canvas_rgba[a +10] as i32 - target_rgba[b +10] as i32;

            let dr3 = canvas_rgba[a +12] as i32 - target_rgba[b +12] as i32;
            let dg3 = canvas_rgba[a +13] as i32 - target_rgba[b +13] as i32;
            let db3 = canvas_rgba[a +14] as i32 - target_rgba[b +14] as i32;

            sse = sse.wrapping_add((dr0*dr0 + dg0*dg0 + db0*db0) as u64);
            sse = sse.wrapping_add((dr1*dr1 + dg1*dg1 + db1*db1) as u64);
            sse = sse.wrapping_add((dr2*dr2 + dg2*dg2 + db2*db2) as u64);
            sse = sse.wrapping_add((dr3*dr3 + dg3*dg3 + db3*db3) as u64);

            i += 16;
        }

        // tail
        while i < end {
            let a = base + i;
            let b = base + i;
            let dr = canvas_rgba[a] as i32 - target_rgba[b] as i32;
            let dg = canvas_rgba[a + 1] as i32 - target_rgba[b + 1] as i32;
            let db = canvas_rgba[a + 2] as i32 - target_rgba[b + 2] as i32;
            sse = sse.wrapping_add((dr*dr + dg*dg + db*db) as u64);
            i += 4;
        }
    }
    sse
}

#[inline]
pub(crate) fn total_sse_region_with_cutoff(
    region_rgba: &[u8],
    target_rgba: &[u8],
    canvas_w: usize,
    region: &IntRect,
    cutoff: u64, // bail early if sse > cutoff
) -> u64 {
    let mut sse: u64 = 0;
    for y in 0..region.h {
        let a0 = (y * region.w) * 4;
        let b0 = ((region.y + y) * canvas_w + region.x) * 4;

        let mut i = 0usize;
        let end = region.w * 4;

        while i + 16 <= end {
            let a = a0 + i;
            let b = b0 + i;

            macro_rules! accum4 { ($off:expr) => {{
                let dr = region_rgba[a + $off] as i32 - target_rgba[b + $off] as i32;
                let dg = region_rgba[a + $off + 1] as i32 - target_rgba[b + $off + 1] as i32;
                let db = region_rgba[a + $off + 2] as i32 - target_rgba[b + $off + 2] as i32;
                (dr*dr + dg*dg + db*db) as u64
            }}}

            sse = sse.wrapping_add(accum4!(0));
            sse = sse.wrapping_add(accum4!(4));
            sse = sse.wrapping_add(accum4!(8));
            sse = sse.wrapping_add(accum4!(12));

            if sse > cutoff { return sse; }
            i += 16;
        }

        while i < end {
            let a = a0 + i;
            let b = b0 + i;
            let dr = region_rgba[a] as i32 - target_rgba[b] as i32;
            let dg = region_rgba[a + 1] as i32 - target_rgba[b + 1] as i32;
            let db = region_rgba[a + 2] as i32 - target_rgba[b + 2] as i32;
            sse = sse.wrapping_add((dr*dr + dg*dg + db*db) as u64);
            if sse > cutoff { return sse; }
            i += 4;
        }
    }
    sse
}

/// Draw ONLY the triangles that overlap `region`, using the spatial index to cull.
/// For the changed triangle, we draw the *candidate* version; all others are from
/// the current DNA. The region buffer must be cleared to background before calling.
pub(crate) fn compose_candidate_region_with_culled_subset(
    out_region_rgba: &mut [u8],
    region: &IntRect,
    canvas_w: usize,
    canvas_h: usize,
    current_dna: &TriangleDna,
    spatial_index: &mut TriangleSpatialIndex,
    changed_index: Option<usize>,
    candidate_dna: &TriangleDna,
) {
    // Triangles potentially affecting this region (sorted indices → draw order)
    let tri_count_temp = current_dna.triangles.len();
    let mut indices = spatial_index.triangles_overlapping_region(region, canvas_w, canvas_h, tri_count_temp);

    // Ensure the changed triangle is included (in case it moved into the region)
    if let Some(ci) = changed_index {
        if !indices.iter().any(|&v| v == ci) {
            indices.push(ci);
            indices.sort_unstable();
        }
    }

    for idx in indices {
        let tri = if Some(idx) == changed_index {
            &candidate_dna.triangles[idx]
        } else {
            &current_dna.triangles[idx]
        };
        rasterize_triangle_over_unpremul_rgba_clipped_region(out_region_rgba, region, canvas_w, canvas_h, tri);
    }
}