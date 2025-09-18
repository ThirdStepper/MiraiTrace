// -----------------------------------------------------------------------------
// Tile grid (error cache partition)
// -----------------------------------------------------------------------------

use rayon::prelude::*;

use super::IntRect;

#[derive(Clone, Debug)]
pub(crate) struct TileGrid {
    pub(crate) tile_size: usize,
    pub(crate) tiles_x: usize,
    pub(crate) tiles_y: usize,
    /// Sum of squared errors (SSE) for each tile (RGB only).
    pub(crate) sse_per_tile: Vec<u64>,
}

impl TileGrid {
    pub(crate) fn new(canvas_w: usize, canvas_h: usize, tile_size: usize) -> Self {
        let tiles_x = (canvas_w + tile_size - 1) / tile_size;
        let tiles_y = (canvas_h + tile_size - 1) / tile_size;
        Self {
            tile_size,
            tiles_x,
            tiles_y,
            sse_per_tile: vec![0; tiles_x * tiles_y],
        }
    }

    #[inline]
    pub(crate) fn tile_index(&self, tx: usize, ty: usize) -> usize {
        debug_assert!(self.tiles_x > 0 && self.tiles_y > 0);
        debug_assert!(
            tx < self.tiles_x && ty < self.tiles_y,
            "tile_index oob: tx={}, ty={}, tiles_x={}, tiles_y={}",
            tx, ty, self.tiles_x, self.tiles_y
        );
        ty * self.tiles_x + tx
    }

    #[inline]
    pub(crate) fn blit_region_and_update_sse(
        &mut self,
        canvas_rgba: &mut [u8],
        target_rgba: &[u8],
        canvas_w: usize,
        region: &IntRect,   // { x, y, w, h }
        region_rgba: &[u8], // tightly packed: stride = region.w * 4
    ) {
        if region.w == 0 || region.h == 0 { return; }
        debug_assert_eq!(canvas_rgba.len(), target_rgba.len());
        if canvas_w == 0 { return; }

        // Canvas geometry
        let px_total  = canvas_rgba.len() / 4;
        let canvas_h  = px_total / canvas_w;
        let row_stride = canvas_w * 4;

        // Source geometry (ALWAYS use region.w/region.h for src indexing)
        let src_w = region.w;
        let src_h = region.h;
        let src_row_bytes = src_w * 4;

        // Validate source buffer against the declared region size
        let needed = src_h.saturating_mul(src_row_bytes);
        if region_rgba.len() < needed {
            debug_assert!(
                region_rgba.len() >= needed,
                "region_rgba too small: have {}, need {} ({}x{} region)",
                region_rgba.len(), needed, src_w, src_h
            );
            return;
        }

        let tile_size = self.tile_size;
        let tiles_x   = self.tiles_x;
        let tiles_y   = self.tiles_y;

        // For each source row ry, compute its destination y and clip to canvas
        for ry in 0..src_h {
            let y = region.y + ry;
            if y >= canvas_h { break; }                // vertical clip (top/bottom)
            let canvas_row = y * row_stride;
            let src_row    = ry * src_row_bytes;

            // Horizontal clipping: [cx0, cx1) on canvas corresponds to
            //                     [sx0, sx1) on source (same width)
            let cx0 = region.x;
            if cx0 >= canvas_w { continue; }           // fully off to the right
            let cx1 = (region.x + src_w).min(canvas_w);
            if cx0 >= cx1 { continue; }                // empty after clipping

            // Map to source x range
            let sx0 = cx0 - region.x;                  // 0 .. src_w-1
            //let sx1 = cx1 - region.x;

            // Walk spans that don't cross tile boundaries on the canvas
            // (tile membership is based on canvas coords)
            let ty = (y / tile_size).min(tiles_y.saturating_sub(1));

            let mut x = cx0;
            while x < cx1 {
                let tx = (x / tile_size).min(tiles_x.saturating_sub(1));
                let tile_idx = ty * tiles_x + tx;

                // span end = min(end of this tile, end of clipped row)
                let span_end_x = ((tx + 1) * tile_size).min(cx1);

                // Byte offsets
                let mut c = canvas_row + x * 4;                        // canvas byte index
                let mut i = src_row + (x - cx0 + sx0) * 4;             // source byte index
                let end_i = src_row + (span_end_x - cx0 + sx0) * 4;    // exclusive

                // Accumulate SSE delta across the span, apply once
                let mut span_delta: i64 = 0;

                // Bulk: 4 pixels
                while i + 16 <= end_i {
                    // p0
                    {
                        let c0 = c; let i0 = i;
                        let dr0 = canvas_rgba[c0] as i32     - target_rgba[c0] as i32;
                        let dg0 = canvas_rgba[c0 + 1] as i32 - target_rgba[c0 + 1] as i32;
                        let db0 = canvas_rgba[c0 + 2] as i32 - target_rgba[c0 + 2] as i32;
                        let old_err = (dr0*dr0 + dg0*dg0 + db0*db0) as i64;

                        let dr1 = region_rgba[i0] as i32     - target_rgba[c0] as i32;
                        let dg1 = region_rgba[i0 + 1] as i32 - target_rgba[c0 + 1] as i32;
                        let db1 = region_rgba[i0 + 2] as i32 - target_rgba[c0 + 2] as i32;
                        let new_err = (dr1*dr1 + dg1*dg1 + db1*db1) as i64;

                        span_delta += new_err - old_err;

                        canvas_rgba[c0]     = region_rgba[i0];
                        canvas_rgba[c0 + 1] = region_rgba[i0 + 1];
                        canvas_rgba[c0 + 2] = region_rgba[i0 + 2];
                        canvas_rgba[c0 + 3] = region_rgba[i0 + 3];
                    }
                    // p1
                    {
                        let c1 = c + 4; let i1 = i + 4;
                        let dr0 = canvas_rgba[c1] as i32     - target_rgba[c1] as i32;
                        let dg0 = canvas_rgba[c1 + 1] as i32 - target_rgba[c1 + 1] as i32;
                        let db0 = canvas_rgba[c1 + 2] as i32 - target_rgba[c1 + 2] as i32;
                        let old_err = (dr0*dr0 + dg0*dg0 + db0*db0) as i64;

                        let dr1 = region_rgba[i1] as i32     - target_rgba[c1] as i32;
                        let dg1 = region_rgba[i1 + 1] as i32 - target_rgba[c1 + 1] as i32;
                        let db1 = region_rgba[i1 + 2] as i32 - target_rgba[c1 + 2] as i32;
                        let new_err = (dr1*dr1 + dg1*dg1 + db1*db1) as i64;

                        span_delta += new_err - old_err;

                        canvas_rgba[c1]     = region_rgba[i1];
                        canvas_rgba[c1 + 1] = region_rgba[i1 + 1];
                        canvas_rgba[c1 + 2] = region_rgba[i1 + 2];
                        canvas_rgba[c1 + 3] = region_rgba[i1 + 3];
                    }
                    // p2
                    {
                        let c2 = c + 8; let i2 = i + 8;
                        let dr0 = canvas_rgba[c2] as i32     - target_rgba[c2] as i32;
                        let dg0 = canvas_rgba[c2 + 1] as i32 - target_rgba[c2 + 1] as i32;
                        let db0 = canvas_rgba[c2 + 2] as i32 - target_rgba[c2 + 2] as i32;
                        let old_err = (dr0*dr0 + dg0*dg0 + db0*db0) as i64;

                        let dr1 = region_rgba[i2] as i32     - target_rgba[c2] as i32;
                        let dg1 = region_rgba[i2 + 1] as i32 - target_rgba[c2 + 1] as i32;
                        let db1 = region_rgba[i2 + 2] as i32 - target_rgba[c2 + 2] as i32;
                        let new_err = (dr1*dr1 + dg1*dg1 + db1*db1) as i64;

                        span_delta += new_err - old_err;

                        canvas_rgba[c2]     = region_rgba[i2];
                        canvas_rgba[c2 + 1] = region_rgba[i2 + 1];
                        canvas_rgba[c2 + 2] = region_rgba[i2 + 2];
                        canvas_rgba[c2 + 3] = region_rgba[i2 + 3];
                    }
                    // p3
                    {
                        let c3 = c + 12; let i3 = i + 12;
                        let dr0 = canvas_rgba[c3] as i32     - target_rgba[c3] as i32;
                        let dg0 = canvas_rgba[c3 + 1] as i32 - target_rgba[c3 + 1] as i32;
                        let db0 = canvas_rgba[c3 + 2] as i32 - target_rgba[c3 + 2] as i32;
                        let old_err = (dr0*dr0 + dg0*dg0 + db0*db0) as i64;

                        let dr1 = region_rgba[i3] as i32     - target_rgba[c3] as i32;
                        let dg1 = region_rgba[i3 + 1] as i32 - target_rgba[c3 + 1] as i32;
                        let db1 = region_rgba[i3 + 2] as i32 - target_rgba[c3 + 2] as i32;
                        let new_err = (dr1*dr1 + dg1*dg1 + db1*db1) as i64;

                        span_delta += new_err - old_err;

                        canvas_rgba[c3]     = region_rgba[i3];
                        canvas_rgba[c3 + 1] = region_rgba[i3 + 1];
                        canvas_rgba[c3 + 2] = region_rgba[i3 + 2];
                        canvas_rgba[c3 + 3] = region_rgba[i3 + 3];
                    }

                    c += 16;
                    i += 16;
                }

                // Tail
                while i < end_i {
                    let dr0 = canvas_rgba[c] as i32     - target_rgba[c] as i32;
                    let dg0 = canvas_rgba[c + 1] as i32 - target_rgba[c + 1] as i32;
                    let db0 = canvas_rgba[c + 2] as i32 - target_rgba[c + 2] as i32;
                    let old_err = (dr0*dr0 + dg0*dg0 + db0*db0) as i64;

                    let dr1 = region_rgba[i] as i32     - target_rgba[c] as i32;
                    let dg1 = region_rgba[i + 1] as i32 - target_rgba[c + 1] as i32;
                    let db1 = region_rgba[i + 2] as i32 - target_rgba[c + 2] as i32;
                    let new_err = (dr1*dr1 + dg1*dg1 + db1*db1) as i64;

                    span_delta += new_err - old_err;

                    canvas_rgba[c]     = region_rgba[i];
                    canvas_rgba[c + 1] = region_rgba[i + 1];
                    canvas_rgba[c + 2] = region_rgba[i + 2];
                    canvas_rgba[c + 3] = region_rgba[i + 3];

                    c += 4;
                    i += 4;
                }

                // Apply the delta once for this span
                if let Some(s) = self.sse_per_tile.get_mut(tile_idx) {
                    let cur = (*s as i64).saturating_add(span_delta);
                    *s = cur.max(0) as u64;
                }

                x = span_end_x;
            }
        }
    }




    /// Return all tiles overlapped by `rect_px`.
    pub(crate) fn tiles_overlapping_rect(
        &self,
        r: &IntRect,
        canvas_w: usize,
        canvas_h: usize,
    ) -> Vec<(usize, usize)> {
        if r.w == 0 || r.h == 0 { return Vec::new(); }

        // Clip rect to canvas (pixel coords, x1/y1 are exclusive)
        let x0 = r.x.min(canvas_w);
        let y0 = r.y.min(canvas_h);
        let x1 = (r.x + r.w).min(canvas_w);
        let y1 = (r.y + r.h).min(canvas_h);
        if x0 >= x1 || y0 >= y1 { return Vec::new(); }

        // Map [x0, x1) to tile range [tx0 ..= tx1], same for y
        let tx0 = (x0 / self.tile_size).min(self.tiles_x.saturating_sub(1));
        let ty0 = (y0 / self.tile_size).min(self.tiles_y.saturating_sub(1));
        let tx1 = ((x1 - 1) / self.tile_size).min(self.tiles_x.saturating_sub(1));
        let ty1 = ((y1 - 1) / self.tile_size).min(self.tiles_y.saturating_sub(1));

        let mut out = Vec::with_capacity((tx1 - tx0 + 1) * (ty1 - ty0 + 1));
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                out.push((tx, ty));
            }
        }
        out
    }

    /// Compute a tight region (in pixels) that covers a set of tiles.
    #[allow(dead_code)]
    pub(crate) fn union_rect_for_tiles(
        &self,
        tiles: &[(usize, usize)],
        canvas_w: usize,
        canvas_h: usize,
    ) -> IntRect {
        if tiles.is_empty() {
            return IntRect { x: 0, y: 0, w: 0, h: 0 };
        }

        let mut tx_min = self.tiles_x - 1;
        let mut ty_min = self.tiles_y - 1;
        let mut tx_max = 0usize;
        let mut ty_max = 0usize;
        for &(tx, ty) in tiles {
            tx_min = tx_min.min(tx);
            ty_min = ty_min.min(ty);
            tx_max = tx_max.max(tx);
            ty_max = ty_max.max(ty);
        }

        let x  = tx_min * self.tile_size;
        let y  = ty_min * self.tile_size;
        let xe = ((tx_max + 1) * self.tile_size).min(canvas_w); // exclusive
        let ye = ((ty_max + 1) * self.tile_size).min(canvas_h); // exclusive
        IntRect { x, y, w: xe - x, h: ye - y }
    }

    #[allow(dead_code)]
    pub fn sum_error_for_tiles(&self, tiles: &[(usize, usize)]) -> u64 {
        tiles
            .iter()
            .map(|&(tx, ty)| self.sse_per_tile[self.tile_index(tx, ty)])
            .sum()
    }

    /// After accepting a candidate, recompute SSE for each affected tile by comparing
    /// current canvas vs target. This touches only the small set of tiles provided.
    pub fn recompute_tiles_sse_from_canvas(
        &mut self,
        tiles: &[(usize, usize)],
        canvas_rgba: &[u8],
        target_rgba: &[u8],
        canvas_w: usize,
        canvas_h: usize,
    ) {
        // Compute SSE for each tile in parallel, then commit results.
        let tile_size = self.tile_size;
        let updates: Vec<(usize, u64)> = tiles
            .par_iter()
            .filter_map(|&(tx, ty)| {
                // Defensive: drop invalid tiles
                if tx >= self.tiles_x || ty >= self.tiles_y {
                    debug_assert!(
                        false,
                        "recompute_tiles_sse_from_canvas: oob tile (tx={}, ty={}) for grid {}x{}",
                        tx, ty, self.tiles_x, self.tiles_y
                    );
                    return None;
                }
            
                let px0 = tx * tile_size;
                let py0 = ty * tile_size;
                let px1 = (px0 + tile_size).min(canvas_w);
                let py1 = (py0 + tile_size).min(canvas_h);
            
                let mut sse: u64 = 0;
                for y in py0..py1 {
                    let row_off = y * canvas_w * 4;
                    let mut i = row_off + px0 * 4;
                    let row_end = row_off + px1 * 4;
                
                    while i + 16 <= row_end {
                        // unroll 4 pixels
                        let dr0 = canvas_rgba[i] as i32     - target_rgba[i] as i32;
                        let dg0 = canvas_rgba[i + 1] as i32 - target_rgba[i + 1] as i32;
                        let db0 = canvas_rgba[i + 2] as i32 - target_rgba[i + 2] as i32;
                    
                        let j = i + 4;
                        let dr1 = canvas_rgba[j] as i32     - target_rgba[j] as i32;
                        let dg1 = canvas_rgba[j + 1] as i32 - target_rgba[j + 1] as i32;
                        let db1 = canvas_rgba[j + 2] as i32 - target_rgba[j + 2] as i32;
                    
                        let k = i + 8;
                        let dr2 = canvas_rgba[k] as i32     - target_rgba[k] as i32;
                        let dg2 = canvas_rgba[k + 1] as i32 - target_rgba[k + 1] as i32;
                        let db2 = canvas_rgba[k + 2] as i32 - target_rgba[k + 2] as i32;
                    
                        let m = i + 12;
                        let dr3 = canvas_rgba[m] as i32     - target_rgba[m] as i32;
                        let dg3 = canvas_rgba[m + 1] as i32 - target_rgba[m + 1] as i32;
                        let db3 = canvas_rgba[m + 2] as i32 - target_rgba[m + 2] as i32;
                    
                        sse += (dr0*dr0 + dg0*dg0 + db0*db0) as u64;
                        sse += (dr1*dr1 + dg1*dg1 + db1*db1) as u64;
                        sse += (dr2*dr2 + dg2*dg2 + db2*db2) as u64;
                        sse += (dr3*dr3 + dg3*dg3 + db3*db3) as u64;
                    
                        i += 16;
                    }
                    while i < row_end {
                        let dr = canvas_rgba[i] as i32     - target_rgba[i] as i32;
                        let dg = canvas_rgba[i + 1] as i32 - target_rgba[i + 1] as i32;
                        let db = canvas_rgba[i + 2] as i32 - target_rgba[i + 2] as i32;
                        sse += (dr*dr + dg*dg + db*db) as u64;
                        i += 4;
                    }
                }
            
                Some((self.tile_index(tx, ty), sse))
            })
            .collect();

        for (tile_idx, sse) in updates {
            self.sse_per_tile[tile_idx] = sse;
        }
    }
}

#[inline]
pub(crate) fn choose_tile_size(w: usize, h: usize) -> usize {
    let px = w.saturating_mul(h);
    if px >= 2560  * 1440 { 96 }       // ~1440p+
    else if px >= 1920 * 1080 { 64 }  // 1080p+
    else { 32 }
}