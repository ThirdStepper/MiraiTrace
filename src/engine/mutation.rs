// -----------------------------------------------------------------------------
// Mutation operators
// -----------------------------------------------------------------------------

use rand::prelude::*;
use rand_pcg::Pcg64Mcg as PcgRng;

use super::{IntRect, TileGrid, Triangle, TriangleDna, triangle_bbox_px};
use super::{clamp_i32, union_rect};

// ---------- Painterly helpers ----------
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Progress in [0,1] based on how many triangles we already have compared to a goal.
/// `tiles_total` = tile_grid.tiles_x * tile_grid.tiles_y
#[inline]
fn growth_phase(tri_count: usize, tiles_total: usize, triangles_per_tile_goal: usize) -> f32 {
    let goal = (tiles_total * triangles_per_tile_goal).max(1) as f32;
    (tri_count as f32 / goal).clamp(0.0, 1.0)
}

/// Spawn radius (in pixels) for a NEW triangle:
/// Early phase → big strokes; late phase → small, detail strokes.
#[inline]
fn spawn_radius_for_phase(phase: f32, canvas_w: usize, canvas_h: usize, tile_size_px: usize) -> f32 {
    let max_dim = canvas_w.max(canvas_h) as f32;
    let big = max_dim * 0.22;                   // large early strokes (≈ 22% of long side)
    let small = (tile_size_px as f32) * 0.45;   // late strokes ≈ half a tile
    lerp(big, small, phase)
}

/// New-triangle opacity schedule (slightly more opaque early, glaze later).
#[inline]
fn alpha_for_phase(phase: f32) -> u8 {
    // map 0.0..1.0 → 180..120
    let a = lerp(180.0, 120.0, phase);
    a.round().clamp(0.0, 255.0) as u8
}

/// MoveVertex jitter in pixels. Big moves early, tiny nudges late.
#[inline]
fn move_vertex_jitter_for_phase(phase: f32) -> i32 {
    // map 0.0..1.0 → 12..3 px
    let px = lerp(12.0, 3.0, phase);
    px.round().clamp(1.0, 64.0) as i32
}


// ---------- Mutation Enum -------------
#[derive(Clone, Copy, Debug)]
pub enum MutationOperator {
    AddTriangle,
    RemoveTriangle,
    MoveVertex,
    Recolor,
}

impl MutationOperator {
    pub const COUNT: usize = 4;
    pub fn index(self) -> usize {
        match self {
            MutationOperator::AddTriangle => 0,
            MutationOperator::RemoveTriangle => 1,
            MutationOperator::MoveVertex => 2,
            MutationOperator::Recolor => 3,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            MutationOperator::AddTriangle => "Add",
            MutationOperator::RemoveTriangle => "Remove",
            MutationOperator::MoveVertex => "Move",
            MutationOperator::Recolor => "Recolor",
        }
    }
}

// ---------- Mutation Proposal Type -------------
/// A candidate mutation with metadata used for fast evaluation/commit.

#[derive(Clone)]
pub(super) struct Proposal {
    /// The candidate DNA after applying the mutation.
    pub(super) candidate_dna_out: TriangleDna,
    /// Conservative pixel-space bbox that covers what changed.
    pub(super) affected_bbox_px: IntRect,
    /// Which draw index changed/was added (if any).
    pub(super) changed_index: Option<usize>,
    /// For MoveVertex: previous geometry (so we can update the index).
    pub(super) old_triangle_for_update: Option<Triangle>,
    /// Which operator produced this proposal.
    #[allow(dead_code)]
    pub(super) op: MutationOperator,
}

// ---------- Local helpers used by proposer ----------

/// Pick a tile with probability proportional to its current SSE (higher error → more likely).
fn pick_error_weighted_tile(tile_grid: &TileGrid, rng: &mut PcgRng) -> (usize, usize) {
    // Sum total error
    let mut total: u128 = 0;
    for ty in 0..tile_grid.tiles_y {
        for tx in 0..tile_grid.tiles_x {
            total += tile_grid.sse_per_tile[tile_grid.tile_index(tx, ty)] as u128;
        }
    }
    // If everything is zero (e.g., blank target), just pick random
    if total == 0 {
        let tx = rng.gen_range(0..tile_grid.tiles_x);
        let ty = rng.gen_range(0..tile_grid.tiles_y);
        return (tx, ty);
    }
    // Roulette-wheel
    let mut r = rng.gen_range(0..=total - 1);
    for ty in 0..tile_grid.tiles_y {
        for tx in 0..tile_grid.tiles_x {
            let w = tile_grid.sse_per_tile[tile_grid.tile_index(tx, ty)] as u128;
            if r < w {
                return (tx, ty);
            }
            r -= w;
        }
    }
    (tile_grid.tiles_x - 1, tile_grid.tiles_y - 1)
}

/// Take a representative color from the target inside a tile (average of K random pixels).
/// Samples more pixels for better color accuracy.
fn sample_average_target_color_in_tile(
    target_rgba: &[u8],
    canvas_w: usize,
    canvas_h: usize,
    tile_grid: &TileGrid,
    tile_xy: (usize, usize),
    tri_count: usize,
    rng: &mut PcgRng,
) -> [u8; 4] {
    let (tx, ty) = tile_xy;
    let x0 = tx * tile_grid.tile_size;
    let y0 = ty * tile_grid.tile_size;
    let x1 = (x0 + tile_grid.tile_size).min(canvas_w);
    let y1 = (y0 + tile_grid.tile_size).min(canvas_h);

    let mut acc_r: u64 = 0;
    let mut acc_g: u64 = 0;
    let mut acc_b: u64 = 0;

    // Increased sample count from 16 to 64 for better color accuracy
    let samples = 64.min(((x1 - x0) * (y1 - y0)).max(1));
    for _ in 0..samples {
        let x = rng.gen_range(x0..x1);
        let y = rng.gen_range(y0..y1);
        let i = (y * canvas_w + x) * 4;
        acc_r += target_rgba[i] as u64;
        acc_g += target_rgba[i + 1] as u64;
        acc_b += target_rgba[i + 2] as u64;
    }
    let r = (acc_r / samples as u64) as u8;
    let g = (acc_g / samples as u64) as u8;
    let b = (acc_b / samples as u64) as u8;

    // Painterly alpha: early = stronger coverage, later = gentle glaze
    let tiles_total = tile_grid.tiles_x * tile_grid.tiles_y;
    let phase = growth_phase(tri_count, tiles_total, 8);
    let a = alpha_for_phase(phase);

    // Add slight jitter to alpha to create more varied opacity (±10%)
    let alpha_jitter = rng.gen_range(-25..=25);
    let a_jittered = (a as i32 + alpha_jitter).clamp(30, 220) as u8;

    [r, g, b, a_jittered]
}

/// Create a triangle roughly centered in the given tile, with vertices scattered
/// on a circle around the center. Radius tapers from “big strokes” early to
/// “small detail” late based on the current genome size.
fn triangle_centered_in_tile(
    canvas_w: usize,
    canvas_h: usize,
    tile_grid: &TileGrid,
    tile_xy: (usize, usize),
    color_rgba: [u8; 4],
    tri_count: usize,        // ← ADDED
    rng: &mut PcgRng,
) -> Triangle {
    let (tx, ty) = tile_xy;

    // Spawn center: tile center (feel free to randomize in-tile if you prefer)
    let cx = (tx * tile_grid.tile_size + tile_grid.tile_size / 2) as f32;
    let cy = (ty * tile_grid.tile_size + tile_grid.tile_size / 2) as f32;

    // Painterly spawn radius
    let tiles_total = tile_grid.tiles_x * tile_grid.tiles_y;
    let phase = growth_phase(tri_count, tiles_total, 8);
    let radius = spawn_radius_for_phase(
        phase,
        canvas_w,
        canvas_h,
        tile_grid.tile_size,
    );

    // Build three vertices around the center with jittered angles & radii
    let mut vx = [0i32; 3];
    let mut vy = [0i32; 3];
    for i in 0..3 {
        let theta: f32 = rng.gen_range(0.0..(2.0 * std::f32::consts::PI));
        // Jitter radius by ~±40% for variety
        let r = radius * rng.gen_range(0.6..1.4);
        let px = (cx + r * theta.cos()).round() as i32;
        let py = (cy + r * theta.sin()).round() as i32;
        vx[i] = clamp_i32(px, 0, canvas_w as i32 - 1);
        vy[i] = clamp_i32(py, 0, canvas_h as i32 - 1);
    }

    Triangle {
        x0: vx[0], y0: vy[0],
        x1: vx[1], y1: vy[1],
        x2: vx[2], y2: vy[2],
        r: color_rgba[0],
        g: color_rgba[1],
        b: color_rgba[2],
        a: color_rgba[3], // alpha already scheduled by the color sampler
    }
}

// ---------- Micro-optimization (Evolve-style refinement) ----------

/// Generate small variations of an accepted mutation for local refinement.
/// This mimics tux3/Evolve's aggressive post-acceptance optimization.
/// Returns a batch of micro-proposals that make small tweaks to the accepted triangle.
pub(super) fn generate_micro_variations(
    current_dna: &TriangleDna,
    changed_index: usize,
    canvas_w: i32,
    canvas_h: i32,
    rng: &mut PcgRng,
    num_variations: usize,
) -> Vec<Proposal> {
    if changed_index >= current_dna.triangles.len() {
        return Vec::new();
    }

    let mut variations = Vec::with_capacity(num_variations);
    let original_tri = &current_dna.triangles[changed_index];

    // Generate micro-variations:
    // 1. Color variations (±5 in RGB)
    // 2. Alpha variations (±10)
    // 3. Vertex micro-movements (±2-3 pixels)

    for i in 0..num_variations {
        let mut dna_variant = current_dna.clone();
        let tri = &mut dna_variant.triangles[changed_index];

        match i % 3 {
            0 => {
                // Color variation: adjust one random channel
                let channel = rng.gen_range(0..3);
                let delta = rng.gen_range(-5..=5);
                match channel {
                    0 => tri.r = (tri.r as i32 + delta).clamp(0, 255) as u8,
                    1 => tri.g = (tri.g as i32 + delta).clamp(0, 255) as u8,
                    _ => tri.b = (tri.b as i32 + delta).clamp(0, 255) as u8,
                }
            }
            1 => {
                // Alpha variation
                let delta = rng.gen_range(-10..=10);
                tri.a = (tri.a as i32 + delta).clamp(30, 220) as u8;
            }
            _ => {
                // Vertex micro-movement: move one random vertex slightly
                let vertex = rng.gen_range(0..3);
                let dx = rng.gen_range(-2..=2);
                let dy = rng.gen_range(-2..=2);

                match vertex {
                    0 => {
                        tri.x0 = clamp_i32(tri.x0 + dx, 0, canvas_w - 1);
                        tri.y0 = clamp_i32(tri.y0 + dy, 0, canvas_h - 1);
                    }
                    1 => {
                        tri.x1 = clamp_i32(tri.x1 + dx, 0, canvas_w - 1);
                        tri.y1 = clamp_i32(tri.y1 + dy, 0, canvas_h - 1);
                    }
                    _ => {
                        tri.x2 = clamp_i32(tri.x2 + dx, 0, canvas_w - 1);
                        tri.y2 = clamp_i32(tri.y2 + dy, 0, canvas_h - 1);
                    }
                }
            }
        }

        // Conservative affected bbox = union(old, new)
        let bbox_old = triangle_bbox_px(original_tri, canvas_w as usize, canvas_h as usize);
        let bbox_new = triangle_bbox_px(tri, canvas_w as usize, canvas_h as usize);
        let affected = union_rect(&bbox_old, &bbox_new);

        variations.push(Proposal {
            candidate_dna_out: dna_variant,
            affected_bbox_px: affected,
            changed_index: Some(changed_index),
            old_triangle_for_update: Some(original_tri.clone()),
            op: MutationOperator::Recolor, // Micro-optimizations are refinements
        });
    }

    variations
}

// ---------- Main entry called by engine_core ----------

pub(super) fn propose_mutation_with_bbox(
    current_dna: &TriangleDna,
    rng: &mut PcgRng,
    canvas_w: i32,
    canvas_h: i32,
    max_tris: usize,
    op: MutationOperator,
    tile_grid: &TileGrid,
    target_rgba: &[u8],
) -> Proposal {
    match op {
        MutationOperator::AddTriangle => {
            let mut dna2 = current_dna.clone();

            if dna2.triangles.len() >= max_tris {
                // Fallback: recolor instead (keeps compile-time behavior simple)
                return propose_mutation_with_bbox(
                    current_dna, rng, canvas_w, canvas_h, max_tris,
                    MutationOperator::Recolor, tile_grid, target_rgba,
                );
            }

            // 1) Pick a tile where the error is high.
            let chosen_tile = pick_error_weighted_tile(tile_grid, rng);

            // 2) Pull a representative color from the target inside that tile (with painterly alpha).
            let avg_color = sample_average_target_color_in_tile(
                target_rgba,
                canvas_w as usize,
                canvas_h as usize,
                tile_grid,
                chosen_tile,
                dna2.triangles.len(), // tri_count for scheduling
                rng,
            );

            // 3) Spawn a triangle centered in that tile with painterly radius (big→small).
            let mut tri = triangle_centered_in_tile(
                canvas_w as usize,
                canvas_h as usize,
                tile_grid,
                chosen_tile,
                avg_color,
                dna2.triangles.len(), // tri_count for scheduling
                rng,
            );

            // Add slight color variation to avoid identical triangles (±5 in RGB)
            let color_var = 5i32;
            tri.r = (tri.r as i32 + rng.gen_range(-color_var..=color_var)).clamp(0, 255) as u8;
            tri.g = (tri.g as i32 + rng.gen_range(-color_var..=color_var)).clamp(0, 255) as u8;
            tri.b = (tri.b as i32 + rng.gen_range(-color_var..=color_var)).clamp(0, 255) as u8;

            let new_index = dna2.triangles.len();
            dna2.triangles.push(tri);
            let bbox = triangle_bbox_px(&dna2.triangles[new_index], canvas_w as usize, canvas_h as usize);

            Proposal {
                candidate_dna_out: dna2,
                affected_bbox_px: bbox,
                changed_index: Some(new_index),
                old_triangle_for_update: None,
                op,
            }
        }
        MutationOperator::RemoveTriangle => {
            // Smart removal: remove triangles that likely contribute least
            if current_dna.triangles.is_empty() || current_dna.triangles.len() < 10 {
                // Don't remove if we have too few triangles - add instead
                return propose_mutation_with_bbox(
                    current_dna,
                    rng,
                    canvas_w,
                    canvas_h,
                    max_tris,
                    MutationOperator::AddTriangle,
                    tile_grid,
                    target_rgba,
                );
            }

            // Strategy: Pick triangles that are likely redundant
            // 1. Very low alpha (nearly transparent)
            // 2. Very small area (tiny triangles)
            // 3. In low-error regions (already well-covered)

            let mut candidates = Vec::with_capacity(10);

            // Collect up to 10 candidate triangles for removal
            let sample_size = (current_dna.triangles.len() / 10).clamp(5, 20);
            for _ in 0..sample_size {
                let idx = rng.gen_range(0..current_dna.triangles.len());
                let tri = &current_dna.triangles[idx];

                // Calculate removal priority score (higher = more likely to remove)
                let mut score = 0.0f32;

                // Factor 1: Low alpha = likely not contributing much
                let alpha_factor = 1.0 - (tri.a as f32 / 255.0);
                score += alpha_factor * 2.0; // Weight: 2x

                // Factor 2: Small area = less impact
                let bbox = triangle_bbox_px(tri, canvas_w as usize, canvas_h as usize);
                let area = (bbox.w * bbox.h) as f32;
                let max_area = (canvas_w * canvas_h) as f32;
                let area_factor = 1.0 - (area / max_area).min(1.0);
                score += area_factor * 1.0; // Weight: 1x

                // Factor 3: Check tile error - if tile has low error, triangle might be redundant
                let tile_x = (bbox.x + bbox.w / 2) / tile_grid.tile_size;
                let tile_y = (bbox.y + bbox.h / 2) / tile_grid.tile_size;
                let tile_idx = tile_y.min(tile_grid.tiles_y - 1) * tile_grid.tiles_x + tile_x.min(tile_grid.tiles_x - 1);

                if tile_idx < tile_grid.sse_per_tile.len() {
                    let tile_error = tile_grid.sse_per_tile[tile_idx];
                    // Lower tile error = already well-covered = safe to remove from
                    let max_tile_error = tile_grid.sse_per_tile.iter().copied().max().unwrap_or(1);
                    let error_factor = 1.0 - (tile_error as f32 / max_tile_error.max(1) as f32);
                    score += error_factor * 1.5; // Weight: 1.5x
                }

                candidates.push((idx, score));
            }

            // Sort by score (highest score = best candidate for removal)
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Pick the best candidate (or random if no candidates)
            let idx = if !candidates.is_empty() {
                candidates[0].0
            } else {
                rng.gen_range(0..current_dna.triangles.len())
            };

            let mut dna2 = current_dna.clone();
            let removed = dna2.triangles.remove(idx);
            let bbox = triangle_bbox_px(&removed, canvas_w as usize, canvas_h as usize);

            Proposal {
                candidate_dna_out: dna2,
                affected_bbox_px: bbox,
                changed_index: None,  // No single triangle changed
                old_triangle_for_update: Some(removed),
                op,
            }
        }
        MutationOperator::MoveVertex => {
            if current_dna.triangles.is_empty() {
                return propose_mutation_with_bbox(
                    current_dna,
                    rng,
                    canvas_w,
                    canvas_h,
                    max_tris,
                    MutationOperator::AddTriangle,
                    tile_grid,
                    target_rgba,
                );
            }
            let mut dna2 = current_dna.clone();
            let idx = rng.gen_range(0..dna2.triangles.len());
            let before = dna2.triangles[idx].clone();

            // Systematic directional search: try one of 8 compass directions
            // This is much more efficient than random jitter!
            let tiles_total = tile_grid.tiles_x * tile_grid.tiles_y;
            let phase = growth_phase(dna2.triangles.len(), tiles_total, 8);
            let step_size = move_vertex_jitter_for_phase(phase).max(1); // 1-12 pixels

            // Pick a random vertex and direction
            let which_vertex = rng.gen_range(0..3);
            let which_direction = rng.gen_range(0..8);

            // 8 compass directions: N, NE, E, SE, S, SW, W, NW
            let (dx, dy) = match which_direction {
                0 => (0, -step_size),           // North (up)
                1 => (step_size, -step_size),   // Northeast
                2 => (step_size, 0),            // East (right)
                3 => (step_size, step_size),    // Southeast
                4 => (0, step_size),            // South (down)
                5 => (-step_size, step_size),   // Southwest
                6 => (-step_size, 0),           // West (left)
                _ => (-step_size, -step_size),  // Northwest
            };

            // Apply movement to the selected vertex
            match which_vertex {
                0 => {
                    dna2.triangles[idx].x0 = clamp_i32(dna2.triangles[idx].x0 + dx, 0, canvas_w - 1);
                    dna2.triangles[idx].y0 = clamp_i32(dna2.triangles[idx].y0 + dy, 0, canvas_h - 1);
                }
                1 => {
                    dna2.triangles[idx].x1 = clamp_i32(dna2.triangles[idx].x1 + dx, 0, canvas_w - 1);
                    dna2.triangles[idx].y1 = clamp_i32(dna2.triangles[idx].y1 + dy, 0, canvas_h - 1);
                }
                _ => {
                    dna2.triangles[idx].x2 = clamp_i32(dna2.triangles[idx].x2 + dx, 0, canvas_w - 1);
                    dna2.triangles[idx].y2 = clamp_i32(dna2.triangles[idx].y2 + dy, 0, canvas_h - 1);
                }
            }

            // Conservative affected bbox = union(old, new)
            let bbox_old = triangle_bbox_px(&before, canvas_w as usize, canvas_h as usize);
            let bbox_new = triangle_bbox_px(&dna2.triangles[idx], canvas_w as usize, canvas_h as usize);
            let affected = union_rect(&bbox_old, &bbox_new);

            Proposal {
                candidate_dna_out: dna2,
                affected_bbox_px: affected,
                changed_index: Some(idx),
                old_triangle_for_update: Some(before),
                op,
            }
        }
        MutationOperator::Recolor => {
            if current_dna.triangles.is_empty() {
                return propose_mutation_with_bbox(
                    current_dna,
                    rng,
                    canvas_w,
                    canvas_h,
                    max_tris,
                    MutationOperator::AddTriangle,
                    tile_grid,
                    target_rgba,
                );
            }
            let mut dna2 = current_dna.clone();
            let idx = rng.gen_range(0..dna2.triangles.len());

            // Target-based recolor: Sample actual colors from target image under the triangle
            let tri = &dna2.triangles[idx];
            let bbox = triangle_bbox_px(tri, canvas_w as usize, canvas_h as usize);

            // Sample random pixels within the triangle's bounding box
            let mut acc_r: u64 = 0;
            let mut acc_g: u64 = 0;
            let mut acc_b: u64 = 0;
            let mut sample_count = 0u32;

            let samples = 32.min((bbox.w * bbox.h).max(1));
            for _ in 0..samples {
                let x = rng.gen_range(bbox.x..bbox.x + bbox.w).min(canvas_w as usize - 1);
                let y = rng.gen_range(bbox.y..bbox.y + bbox.h).min(canvas_h as usize - 1);
                let i = (y * canvas_w as usize + x) * 4;

                acc_r += target_rgba[i] as u64;
                acc_g += target_rgba[i + 1] as u64;
                acc_b += target_rgba[i + 2] as u64;
                sample_count += 1;
            }

            if sample_count > 0 {
                // Calculate ideal color from target
                let ideal_r = (acc_r / sample_count as u64) as u8;
                let ideal_g = (acc_g / sample_count as u64) as u8;
                let ideal_b = (acc_b / sample_count as u64) as u8;

                // Interpolate between current and ideal color
                // Use random interpolation factor to create variety
                let interp = rng.gen_range(0.5..1.0); // 50-100% toward ideal

                let t = &mut dna2.triangles[idx];
                t.r = lerp(t.r as f32, ideal_r as f32, interp) as u8;
                t.g = lerp(t.g as f32, ideal_g as f32, interp) as u8;
                t.b = lerp(t.b as f32, ideal_b as f32, interp) as u8;

                // Add small jitter to alpha for variety
                let alpha_jitter = rng.gen_range(-15..=15);
                t.a = (t.a as i32 + alpha_jitter).clamp(30, 220) as u8;
            } else {
                // Fallback: small random jitter if sampling fails
                let mut jitter = |c: u8, range: i32| -> u8 {
                    let n = c as i32 + rng.gen_range(-range..=range);
                    n.clamp(0, 255) as u8
                };
                let t = &mut dna2.triangles[idx];
                t.r = jitter(t.r, 25);
                t.g = jitter(t.g, 25);
                t.b = jitter(t.b, 25);
                t.a = jitter(t.a, 20);
            }

            let bbox = triangle_bbox_px(&dna2.triangles[idx], canvas_w as usize, canvas_h as usize);
            Proposal {
                candidate_dna_out: dna2,
                affected_bbox_px: bbox,
                changed_index: Some(idx),
                old_triangle_for_update: None,
                op,
            }
        }
    }
}