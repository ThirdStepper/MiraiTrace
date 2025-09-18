// -----------------------------------------------------------------------------
// Spatial triangle index
// -----------------------------------------------------------------------------

use super::{Triangle, TriangleDna, IntRect, triangle_bbox_px};

/// Maps tiles → triangle draw indices for fast culling when composing a region.
#[derive(Clone)]
pub(crate) struct TriangleSpatialIndex {
    tile_size: usize,
    tiles_x: usize,
    //tiles_y: usize,  //maybe use this later? not really needed right now though afaict
    /// For each tile, a small vector of draw indices into DNA (ascending order).
    bins: Vec<Vec<usize>>,
    seen_epoch: Vec<u32>, // size >= number of triangles
    epoch: u32,
}



impl TriangleSpatialIndex {
    pub(crate) fn new(canvas_w: usize, canvas_h: usize, tile_size: usize) -> Self {
        let tiles_x = (canvas_w + tile_size - 1) / tile_size;
        let tiles_y = (canvas_h + tile_size - 1) / tile_size;
        Self {
            tile_size,
            tiles_x,
            //tiles_y,
            bins: vec![Vec::new(); tiles_x * tiles_y],
            seen_epoch: Vec::new(),
            epoch: 0,
        }
    }

    #[inline]
    fn bin_index(&self, tx: usize, ty: usize) -> usize {
        ty * self.tiles_x + tx
    }

    #[inline]
    fn ensure_seen_epoch_capacity(&mut self, tri_count: usize) {
        if self.seen_epoch.len() < tri_count {
            self.seen_epoch.resize(tri_count, 0);
        }
    }

    #[inline]
    fn next_epoch(&mut self) -> u32 {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.epoch = 1;
            self.seen_epoch.fill(0);
        }
        self.epoch
    }

    fn tiles_overlapping_rect<'a>(
        &'a self,
        rect_px: &'a IntRect,
        canvas_w: usize,
        canvas_h: usize,
    ) -> impl Iterator<Item = (usize, usize)> + 'a {
        // Clamp rect to canvas
        let x0 = rect_px.x.min(canvas_w.saturating_sub(1));
        let y0 = rect_px.y.min(canvas_h.saturating_sub(1));
        let x1 = (rect_px.x + rect_px.w.saturating_sub(1)).min(canvas_w.saturating_sub(1));
        let y1 = (rect_px.y + rect_px.h.saturating_sub(1)).min(canvas_h.saturating_sub(1));

        let tx0 = x0 / self.tile_size;
        let ty0 = y0 / self.tile_size;
        let tx1 = x1 / self.tile_size;
        let ty1 = y1 / self.tile_size;

        (ty0..=ty1).flat_map(move |ty| (tx0..=tx1).map(move |tx| (tx, ty)))
    }

    /// Rebuild from the full DNA (used at baseline or full reset).
    pub fn rebuild(&mut self, dna: &TriangleDna, canvas_w: usize, canvas_h: usize) {
        for bin in &mut self.bins {
            bin.clear();
        }
        for (draw_index, tri) in dna.triangles.iter().enumerate() {
            let bbox = triangle_bbox_px(tri, canvas_w, canvas_h);
            let overlapped: Vec<_> = self.tiles_overlapping_rect(&bbox, canvas_w, canvas_h).collect();
            for (tx, ty) in overlapped {
                let b = self.bin_index(tx, ty);
                self.bins[b].push(draw_index);
            }
        }
    }

    /// Insert one triangle by draw index & geometry.
    pub fn insert_triangle(&mut self, draw_index: usize, tri: &Triangle, canvas_w: usize, canvas_h: usize) {
        let bbox = triangle_bbox_px(tri, canvas_w, canvas_h);
        let overlapped: Vec<_> = self.tiles_overlapping_rect(&bbox, canvas_w, canvas_h).collect();
        for (tx, ty) in overlapped {
            let b = self.bin_index(tx, ty);
            self.bins[b].push(draw_index);
        }
    }

    /// Remove a triangle from bins using its *old* geometry (linear remove).
    pub fn remove_triangle(&mut self, draw_index: usize, old_triangle: &Triangle, canvas_w: usize, canvas_h: usize) {
        let bbox = triangle_bbox_px(old_triangle, canvas_w, canvas_h);
        let overlapped: Vec<_> = self.tiles_overlapping_rect(&bbox, canvas_w, canvas_h).collect();
        for (tx, ty) in overlapped {
            let b = self.bin_index(tx, ty);
            let bin = &mut self.bins[b];
            if let Some(pos) = bin.iter().position(|&v| v == draw_index) {
                bin.swap_remove(pos);
            }
        }
    }

    /// Fetch a **sorted, de-duplicated** list of draw indices overlapping `region_px`.
    pub fn triangles_overlapping_region(
        &mut self,
        region_px: &IntRect,
        canvas_w: usize,
        canvas_h: usize,
        tri_count: usize,
    ) -> Vec<usize> {
        self.ensure_seen_epoch_capacity(tri_count);
        let tag = self.next_epoch();

        // 1) Materialize tiles to end the immutable borrow before we mutate `self`
        let tiles: Vec<(usize, usize)> = {
            let mut v = Vec::new();
            for t in self.tiles_overlapping_rect(region_px, canvas_w, canvas_h) {
                v.push(t);
            }
            v
        };

        // 2) Now we can mutate `seen_epoch` safely
        let mut out = Vec::with_capacity(64);
        for (tx, ty) in tiles {
            let b = self.bin_index(tx, ty);
            for &idx in &self.bins[b] {
                if idx >= tri_count { continue; }       // safety if bins got stale
                if self.seen_epoch[idx] != tag {
                    self.seen_epoch[idx] = tag;
                    out.push(idx);
                }
            }
        }

        // If draw order != index, sort here by your draw order field
        // out.sort_unstable_by_key(|&i| dna.triangles[i].draw_order);

        out
    }
}