//! Engine core — clean rewrite with consistent error accounting, clear naming, and comments.
//!
//! Responsibilities:
//! - Runs a background worker thread that proposes triangle‑DNA mutations and accepts improvements.
//! - Maintains a per‑tile SSE cache (TileGrid) for fast incremental updates.
//! - Keeps a triangle → tile spatial index to compose only the affected region.
//! - Uses simulated annealing to occasionally accept uphill moves.
//!
//! Key invariants (now enforced):
//! - The scalar total error stored in shared state always equals the **sum over TileGrid tiles**.
//! - Candidate scoring uses **exact rect SSE** for both the current canvas and the candidate buffer.
//! - Best‑of‑K recolor carries the *winning* region + errors forward.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use rand_pcg::Pcg64Mcg as PcgRng;

use super::{
    fill_region_rgba_solid, compose_candidate_region_with_culled_subset,
    total_squared_error_rgb_region_from_buffer_vs_target,
    total_squared_error_rgb_region_from_canvas_vs_target,
    total_sse_region_with_cutoff, pad_and_clamp_rect,
    AdaptiveMutationScheduler, MutationOperator,
    FrameDimensions, IntRect,
    TileGrid, TriangleSpatialIndex, choose_tile_size,
    TriangleDna, SimAnneal,
};
use super::mutation::propose_mutation_with_bbox;
use super::EvolutionStats;

// -----------------------------------------------------------------------------
// Shared state (guarded by a mutex)
// -----------------------------------------------------------------------------
struct SharedState {
    // Canvas & target
    pixel_backbuffer_rgba: Vec<u8>,
    width: usize,
    height: usize,
    target_image_rgba: Option<Vec<u8>>,

    // Colors
    background_color_rgba: [u8; 4],

    // DNA
    dna: TriangleDna,

    // Error scalars (must match tile grid sum)
    current_total_squared_error: Option<u64>,
    best_total_squared_error: Option<u64>,

    // Tile grid & spatial index
    tile_grid: Option<TileGrid>,
    spatial_index: Option<TriangleSpatialIndex>,

    // HUD stats
    stats: EvolutionStats,

    // Counters & pacing
    generation_counter: u64,
    proposals_counter_window: u64,
    accepts_counter_window: u64,
    window_started_at: Instant,

    // uphill-only counters for the current HUD window
    uphill_attempts_counter_window: u64,   // delta > 0
    uphill_accepts_counter_window: u64,    // delta > 0 and accepted

    // runtime dials
    max_triangles_cap: usize,
    work_budget_per_tick: u32,
}
impl Default for SharedState {
    fn default() -> Self {
        Self {
            pixel_backbuffer_rgba: Vec::new(),
            width: 0,
            height: 0,
            target_image_rgba: None,
            background_color_rgba: [0, 0, 0, 255],
            dna: TriangleDna::default(),
            current_total_squared_error: None,
            best_total_squared_error: None,
            tile_grid: None,
            spatial_index: None,
            stats: EvolutionStats { max_triangles_cap: 10_000, ..Default::default() },
            generation_counter: 0,
            proposals_counter_window: 0,
            accepts_counter_window: 0,
            window_started_at: Instant::now(),
            uphill_attempts_counter_window: 0,
            uphill_accepts_counter_window: 0,
            max_triangles_cap: 10_000,
            work_budget_per_tick: 5_000,
        }
    }
}

// -----------------------------------------------------------------------------
// Public engine API (used by UI)
// -----------------------------------------------------------------------------
pub struct EvolutionEngine {
    shared: Arc<Mutex<SharedState>>,
    worker_should_run: Arc<AtomicBool>,
    worker_thread: Option<thread::JoinHandle<()>>,
}

impl EvolutionEngine {
    /// Create the engine, allocate the canvas, and start the worker thread (paused).
    pub fn new(initial_size: FrameDimensions) -> Self {
        let mut shared = SharedState::default();
        shared.width = initial_size.width.max(1);
        shared.height = initial_size.height.max(1);
        let ts = choose_tile_size(shared.width, shared.height);
        shared.pixel_backbuffer_rgba = vec![0; shared.width * shared.height * 4];
        shared.background_color_rgba = [0, 0, 0, 255];
        shared.tile_grid = Some(TileGrid::new(shared.width, shared.height, ts));
        shared.spatial_index = Some(TriangleSpatialIndex::new(shared.width, shared.height, ts));
        shared.window_started_at = Instant::now();

        let mut engine = Self {
            shared: Arc::new(Mutex::new(shared)),
            worker_should_run: Arc::new(AtomicBool::new(false)),
            worker_thread: None,
        };
        engine.start_worker_thread();
        engine
    }

    fn start_worker_thread(&mut self) {
        let shared_mutex = Arc::clone(&self.shared);
        let running_flag = Arc::clone(&self.worker_should_run);

        let handle = thread::spawn(move || {
            let mut rng = PcgRng::new(0x1234_5678_CAFE_BABE_0000_0000_0000_0042u128);
            let mut scheduler = AdaptiveMutationScheduler::new();
            let mut sa = SimAnneal::new();

            // Reused scratch buffer for composing candidate regions
            let mut scratch_region: Vec<u8> = Vec::new();

            loop {
                if !running_flag.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(10));
                    continue;
                }

                // Snapshot canvas constants
                let (canvas_w, canvas_h, bg_rgba, target_opt) = {
                    let s = shared_mutex.lock();
                    (s.width, s.height, s.background_color_rgba, s.target_image_rgba.clone())
                };
                if target_opt.is_none() { thread::sleep(Duration::from_millis(10)); continue; }
                let target_rgba = target_opt.unwrap();

                // -----------------------------------------------------------------
                // Build baseline once (blank canvas vs target) and sync tile grid
                // -----------------------------------------------------------------
                let needs_baseline = { let s = shared_mutex.lock(); s.current_total_squared_error.is_none() };
                if needs_baseline {
                    let full = IntRect { x: 0, y: 0, w: canvas_w, h: canvas_h };

                    scratch_region.resize(full.w * full.h * 4, 0);
                    fill_region_rgba_solid(&mut scratch_region, &full, bg_rgba);

                    #[allow(unused_variables)]
                    let total_err_pixels = total_squared_error_rgb_region_from_buffer_vs_target(
                        &scratch_region, &target_rgba, canvas_w, &full);

                    {
                        let mut s = shared_mutex.lock();
                        let ts = choose_tile_size(s.width, s.height);

                        // Copy the blank image into the live canvas
                        s.pixel_backbuffer_rgba.copy_from_slice(&scratch_region);

                        // Recompute *all* tiles SSE from the canvas so the grid is authoritative
                        let (w, h) = (s.width, s.height);
                        let all_tiles: Vec<(usize, usize)> = {
                            let tg = s.tile_grid.get_or_insert_with(|| TileGrid::new(w, h, ts));
                            (0..tg.tiles_y).flat_map(|ty| (0..tg.tiles_x).map(move |tx| (tx, ty))).collect()
                        };
                        let mut tg = s.tile_grid.take().unwrap_or_else(|| TileGrid::new(w, h, ts));
                        let canvas_ref = &s.pixel_backbuffer_rgba;
                        tg.recompute_tiles_sse_from_canvas(&all_tiles, canvas_ref, &target_rgba, w, h);
                        let tiles_sum = tg.sse_per_tile.iter().copied().sum();

                        #[cfg(debug_assertions)]
                        if tiles_sum != total_err_pixels {
                            eprintln!("Baseline desync: pixel_total={} tiles_sum={}", total_err_pixels, tiles_sum);
                        }

                        // Store canonical totals and grid
                        s.current_total_squared_error = Some(tiles_sum);
                        s.best_total_squared_error = Some(tiles_sum);
                        s.tile_grid = Some(tg);

                        // Spatial index for empty DNA
                        let (w2, h2) = (s.width, s.height);
                        let mut si = s.spatial_index.take().unwrap_or_else(|| TriangleSpatialIndex::new(w2, h2, ts));
                        si.rebuild(&s.dna, w2, h2);
                        s.spatial_index = Some(si);

                        // Re-initialize the annealer on a fresh baseline
                        sa = SimAnneal::new();

                        // ---- EWMA SEED (per-pixel SSE from the baseline) ----
                        let pixels = (s.width as f64) * (s.height as f64);
                        if pixels > 0.0 {
                            // Safe to touch private fields here (same module/file)
                            sa.ppsse_ewma = (tiles_sum as f64 / pixels).max(1.0);
                        }

                        // HUD stats from tiles_sum
                        s.stats.current_error = Some(tiles_sum);
                        s.stats.best_error = Some(tiles_sum);
                        s.stats.last_accept_delta = None;
                        s.stats.error_history.clear();
                        s.stats.push_best_error_history(tiles_sum);
                    }
                    continue; // baseline built; next outer tick
                }

                // How many proposals this tick
                let proposals_this_tick = { let s = shared_mutex.lock(); s.work_budget_per_tick.max(1) };

                for _ in 0..proposals_this_tick {
                    // Snapshot live state needed to score a proposal
                    let (total_err_opt, dna_snapshot, max_tris_cap, grid_snapshot, mut index_snapshot) = {
                        let s = shared_mutex.lock();
                        let ts = choose_tile_size(s.width, s.height);
                        (
                            s.current_total_squared_error,
                            s.dna.clone(),
                            s.max_triangles_cap,
                            s.tile_grid.clone().unwrap_or_else(|| TileGrid::new(s.width, s.height, ts)),
                            s.spatial_index.clone().unwrap_or_else(|| TriangleSpatialIndex::new(s.width, s.height, ts)),
                        )
                    };
                    if total_err_opt.is_none() { break; } // concurrent reset

                    // Always trust the tile grid sum for local computations
                    let current_total_error = grid_snapshot.sse_per_tile.iter().copied().sum::<u64>();

                    // Heuristic for growth target (~8 triangles per tile)
                    let tiles_total = grid_snapshot.tiles_x * grid_snapshot.tiles_y;
                    let growth_target = (tiles_total.saturating_mul(8)).clamp(64, max_tris_cap.max(64));

                    // Choose an operator and propose a mutation guided by tile errors
                    let op = scheduler.sample_operator(&mut rng, dna_snapshot.triangles.len(), growth_target);
                    let mut proposal = propose_mutation_with_bbox(
                        &dna_snapshot, &mut rng,
                        canvas_w as i32, canvas_h as i32,
                        max_tris_cap, op,
                        &grid_snapshot, &target_rgba,
                    );

                    // Geometry-true affected rect (old ⊔ new), *not* tile-aligned
                    let mut union_rect = pad_and_clamp_rect(proposal.affected_bbox_px, 2, canvas_w, canvas_h);
                    if union_rect.w == 0 || union_rect.h == 0 { continue; }
                                    
                    // Now pick tiles to update SSE from this rect (tile-aligned only for SSE)
                    let sse_rect = pad_and_clamp_rect(union_rect, 2, canvas_w, canvas_h);
                    let mut tiles_touched = grid_snapshot.tiles_overlapping_rect(&sse_rect, canvas_w, canvas_h);
                    if tiles_touched.is_empty() { continue; }

                    // Compose candidate buffer for the union rect
                    scratch_region.resize(union_rect.w * union_rect.h * 4, 0);
                    fill_region_rgba_solid(&mut scratch_region, &union_rect, bg_rgba);
                    compose_candidate_region_with_culled_subset(
                        &mut scratch_region, &union_rect, canvas_w, canvas_h,
                        &dna_snapshot, &mut index_snapshot,
                        proposal.changed_index, &proposal.candidate_dna_out,
                    );

                    // Current error over EXACT rect (not whole tiles) — used for cutoff and exact recompute
                    let current_error_in_rect = {
                        let s = shared_mutex.lock();
                        total_squared_error_rgb_region_from_canvas_vs_target(&s.pixel_backbuffer_rgba, &target_rgba, canvas_w, &union_rect)
                    };

                    // Fast candidate estimate with a cutoff for quick rejection
                    let candidate_error_in_rect = total_sse_region_with_cutoff(
                        &scratch_region, &target_rgba, canvas_w, &union_rect, current_error_in_rect);

                    // Combine with checked subtraction
                    let mut candidate_total_error = match current_total_error.checked_sub(current_error_in_rect) {
                        Some(rest) => rest + candidate_error_in_rect,
                        None => {
                            // Should be rare; fall back to grid truth
                            let base = grid_snapshot.sse_per_tile.iter().copied().sum::<u64>().max(current_total_error);
                            base.saturating_sub(current_error_in_rect) + candidate_error_in_rect
                        }
                    };

                    // Best-of-K recolor: try several color variants in the *same* geometry rect
                    if matches!(op, MutationOperator::Recolor) {
                        let k = 4usize;
                        let mut best = (candidate_total_error, proposal.clone(), tiles_touched.clone(), union_rect, candidate_error_in_rect, current_error_in_rect);
                        for _ in 1..k {
                            let p2 = propose_mutation_with_bbox(&dna_snapshot, &mut rng, canvas_w as i32, canvas_h as i32, max_tris_cap, op, &grid_snapshot, &target_rgba);
                            
                            let rect2 = pad_and_clamp_rect(p2.affected_bbox_px, 2, canvas_w, canvas_h);
                            if rect2.w == 0 || rect2.h == 0 { continue; }
                            let tiles2 = grid_snapshot.tiles_overlapping_rect(&sse_rect, canvas_w, canvas_h);
                            if tiles2.is_empty() { continue; }

                            scratch_region.resize(rect2.w * rect2.h * 4, 0);
                            fill_region_rgba_solid(&mut scratch_region, &rect2, bg_rgba);
                            compose_candidate_region_with_culled_subset(
                                &mut scratch_region, &rect2, canvas_w, canvas_h,
                                &dna_snapshot, &mut index_snapshot,
                                p2.changed_index, &p2.candidate_dna_out,
                            );

                            let cur2 = {
                                let s = shared_mutex.lock();
                                total_squared_error_rgb_region_from_canvas_vs_target(&s.pixel_backbuffer_rgba, &target_rgba, canvas_w, &rect2)
                            };

                            let cand2 = total_sse_region_with_cutoff(&scratch_region, &target_rgba, canvas_w, &rect2, cur2);

                            let cand_total2 = match current_total_error.checked_sub(cur2) { Some(rest) => rest + cand2, None => {
                                let base = grid_snapshot.sse_per_tile.iter().copied().sum::<u64>().max(current_total_error);
                                base.saturating_sub(cur2) + cand2
                            }};

                            if cand_total2 < best.0 { best = (cand_total2, p2, tiles2, rect2, cand2, cur2); }
                        }

                        // Adopt winner (region + errors!) and recompose for it
                        //candidate_total_error = best.0;
                        proposal              = best.1;
                        tiles_touched         = best.2;
                        union_rect            = best.3;
                        //candidate_error_in_rect = best.4;
                        //current_error_in_rect   = best.5;

                        scratch_region.resize(union_rect.w * union_rect.h * 4, 0);
                        fill_region_rgba_solid(&mut scratch_region, &union_rect, bg_rgba);
                        compose_candidate_region_with_culled_subset(
                            &mut scratch_region, &union_rect, canvas_w, canvas_h,
                            &dna_snapshot, &mut index_snapshot,
                            proposal.changed_index, &proposal.candidate_dna_out,
                        );
                    }

                    // Final exact recompute (no cutoff) before acceptance decision
                    let exact_current_in_rect = {
                        let s = shared_mutex.lock();
                        total_squared_error_rgb_region_from_canvas_vs_target(&s.pixel_backbuffer_rgba, &target_rgba, canvas_w, &union_rect)
                    };

                    let exact_candidate_in_rect = total_squared_error_rgb_region_from_buffer_vs_target(
                        &scratch_region, &target_rgba, canvas_w, &union_rect);
                        
                    candidate_total_error = match current_total_error.checked_sub(exact_current_in_rect) {
                        Some(rest) => rest + exact_candidate_in_rect,
                        None => {
                            let base = grid_snapshot.sse_per_tile.iter().copied().sum::<u64>().max(current_total_error);
                            base.saturating_sub(exact_current_in_rect) + exact_candidate_in_rect
                        }
                    };

                    // SA acceptance (normalized by region area)
                    
                    let delta = candidate_total_error as i64 - current_total_error as i64;
                    let region_area = (union_rect.w * union_rect.h) as usize;

                    // Phase-aware tightening while below the growth target
                    let tri_count = dna_snapshot.triangles.len();
                    let pre_cap = tri_count < growth_target;
                    sa.set_phase(pre_cap);

                    //let accepted = sa.should_accept(&mut rng, delta, region_area as f64);
                    let accepted = sa.should_accept_area(&mut rng, delta, region_area, exact_current_in_rect);


                    

                    let is_uphill = delta > 0;

                    // Update scheduler bias
                    scheduler.record_outcome(op, accepted);
                    let op_weights = scheduler.current_weights();

                    // Update HUD counters (not touching errors yet)
                    {
                        let mut s = shared_mutex.lock();
                    
                        // Existing global counters
                        s.proposals_counter_window = s.proposals_counter_window.saturating_add(1);
                        s.stats.total_proposals = s.stats.total_proposals.saturating_add(1);
                    
                        // NEW: uphill-only counters (window + lifetime)
                        if is_uphill {
                            s.uphill_attempts_counter_window = s.uphill_attempts_counter_window.saturating_add(1);
                            s.stats.total_uphill_attempts = s.stats.total_uphill_attempts.saturating_add(1);
                            if accepted {
                                s.uphill_accepts_counter_window = s.uphill_accepts_counter_window.saturating_add(1);
                                s.stats.total_uphill_accepts = s.stats.total_uphill_accepts.saturating_add(1);
                            }
                        }
                    
                        // Existing half-second window flush
                        let elapsed = s.window_started_at.elapsed().as_secs_f32();
                        if elapsed >= 0.5 {
                            s.stats.proposals_per_second = s.proposals_counter_window as f32 / elapsed.max(0.001);
                            s.stats.accepts_per_second   = s.accepts_counter_window as f32 / elapsed.max(0.001);
                            s.stats.recent_window_size = (s.proposals_counter_window as usize).max(1);
                            s.stats.recent_acceptance_percent = if s.proposals_counter_window > 0 {
                                (s.accepts_counter_window as f32 / s.proposals_counter_window as f32) * 100.0
                            } else { 0.0 };
                        
                            // NEW: flush uphill-only window metrics into stats
                            s.stats.recent_uphill_window_size = (s.uphill_attempts_counter_window as usize).max(0);
                            s.stats.recent_uphill_acceptance_percent = if s.uphill_attempts_counter_window > 0 {
                                (s.uphill_accepts_counter_window as f32 / s.uphill_attempts_counter_window as f32) * 100.0
                            } else { 0.0 };
                        
                            // reset both sets of window counters
                            s.proposals_counter_window = 0;
                            s.accepts_counter_window = 0;
                            s.uphill_attempts_counter_window = 0;
                            s.uphill_accepts_counter_window = 0;
                        
                            s.window_started_at = Instant::now();
                        }
                    
                        // existing labels/weights
                        s.stats.last_operator_label = Some(op.label().to_string());
                        s.stats.last_tiles_touched = Some(tiles_touched.len());
                        s.stats.operator_weights = vec![
                            ("Add".into(), op_weights[0]),
                            ("Move".into(), op_weights[1]),
                            ("Recolor".into(), op_weights[2]),
                        ];
                        s.stats.max_triangles_cap = s.max_triangles_cap;
                    
                        // keep exporting annealer temp (so HUD can show it)
                        s.stats.anneal_temp = Some(sa.temp()); // or f64 if that’s your field type
                    }

                    if pre_cap {
                        // Peek last window uphill % from a snapshot to avoid holding the lock
                        let (uphill_pct, _) = {
                            let s = shared_mutex.lock();
                            (s.stats.recent_uphill_acceptance_percent, s.window_started_at)
                        };
                        if uphill_pct > 25.0 {
                            sa.nudge_cool(0.96); // small cool-down nudge
                        }
                    }

                    // Rejected → count no-improvement, cool one tick, move on.
                    if !accepted {
                        sa.note_no_improvement();
                        sa.tick();
                        continue;
                    }

                    // Accepted → commit, then evaluate true improvement from tiles_sum and update SA.
                    {
                        let mut s = shared_mutex.lock();
                        let ts = choose_tile_size(s.width, s.height);
                    
                        // Blit + per-tile incremental SSE update
                        let (w, h) = (s.width, s.height);
                        let mut tg = s.tile_grid.take().unwrap_or_else(|| TileGrid::new(w, h, ts));
                        tg.blit_region_and_update_sse(
                            &mut s.pixel_backbuffer_rgba,
                            &target_rgba,
                            canvas_w,
                            &union_rect,
                            &scratch_region,
                        );
                    
                        // Canonical total is the sum over tiles
                        let tiles_sum: u64 = tg.sse_per_tile.iter().copied().sum();
                        #[cfg(debug_assertions)]
                        if tiles_sum != candidate_total_error {
                            eprintln!(
                                "Desync at commit: candidate_total_error={} tiles_sum={}",
                                candidate_total_error, tiles_sum
                            );
                        }
                    
                        // Tell SA if the commit actually improved the global error
                        if tiles_sum < current_total_error {
                            sa.note_improvement();
                        } else {
                            sa.note_no_improvement();
                        }
                        sa.tick();
                    
                        // -------- original commit body (unchanged) --------
                        s.current_total_squared_error = Some(tiles_sum);
                        s.tile_grid = Some(tg);
                    
                        // Update DNA and counters
                        s.dna = proposal.candidate_dna_out.clone();
                        s.stats.triangle_count = s.dna.triangles.len();
                        s.generation_counter = s.generation_counter.saturating_add(1);
                        s.stats.generation_counter = s.generation_counter;
                    
                        // Drive HUD error stats from tiles_sum
                        let prev = s.stats.current_error.unwrap_or(tiles_sum);
                        s.stats.current_error = Some(tiles_sum);
                        s.stats.last_accept_delta = Some(prev as i64 - tiles_sum as i64);
                        if s.best_total_squared_error.map(|b| tiles_sum < b).unwrap_or(true) {
                            s.best_total_squared_error = Some(tiles_sum);
                            s.stats.best_error = Some(tiles_sum);
                            s.stats.push_best_error_history(tiles_sum);
                        }
                    
                        // Spatial index maintenance
                        let (w2, h2) = (s.width, s.height);
                        let mut si = s.spatial_index.take().unwrap_or_else(|| TriangleSpatialIndex::new(w2, h2, ts));
                        match op {
                            MutationOperator::AddTriangle => {
                                if let Some(idx) = proposal.changed_index {
                                    let tri = &s.dna.triangles[idx];
                                    si.insert_triangle(idx, tri, w2, h2);
                                }
                            }
                            MutationOperator::MoveVertex => {
                                if let Some(idx) = proposal.changed_index {
                                    if let Some(ref old_tri) = proposal.old_triangle_for_update {
                                        si.remove_triangle(idx, old_tri, w2, h2);
                                    }
                                    let tri_now = &s.dna.triangles[idx];
                                    si.insert_triangle(idx, tri_now, w2, h2);
                                }
                            }
                            MutationOperator::Recolor => { /* No geometry change */ }
                        }
                        if s.generation_counter % 4096 == 0 {
                            let dna_clone = s.dna.clone();
                            si.rebuild(&dna_clone, w2, h2);
                        }
                        s.spatial_index = Some(si);
                    
                        // Accept counters
                        s.accepts_counter_window = s.accepts_counter_window.saturating_add(1);
                        s.stats.total_accepts = s.stats.total_accepts.saturating_add(1);
                    }
                } // end proposals loop
            } // end worker loop
        });
        self.worker_thread = Some(handle);
    }

    // ---------------------------------------------------------------------
    // Public controls
    // ---------------------------------------------------------------------
    pub fn is_running(&self) -> bool { self.worker_should_run.load(Ordering::Relaxed) }
    pub fn toggle_running_state(&mut self) { let new_state = !self.is_running(); self.worker_should_run.store(new_state, Ordering::Relaxed); }

    pub fn set_max_triangles(&mut self, cap: usize) {
        let mut s = self.shared.lock();
        let clamped = cap.clamp(16, 100_000);
        s.max_triangles_cap = clamped;
        s.stats.max_triangles_cap = clamped;
    }
    pub fn max_triangles(&self) -> usize { let s = self.shared.lock(); s.max_triangles_cap }

    /// Set the clear color and reset the canvas/baseline.
    pub fn set_background_color_and_clear(&mut self, rgba: [u8; 4]) {
        let mut s = self.shared.lock();
        let ts = choose_tile_size(s.width, s.height);
        s.background_color_rgba = rgba;
        s.pixel_backbuffer_rgba.fill(0);
        s.current_total_squared_error = None;
        s.best_total_squared_error = None;
        s.stats = EvolutionStats { max_triangles_cap: s.max_triangles_cap, ..Default::default() };
        s.generation_counter = 0;
        s.tile_grid = Some(TileGrid::new(s.width, s.height, ts));
        s.spatial_index = Some(TriangleSpatialIndex::new(s.width, s.height, ts));
        s.dna.triangles.clear();
    }

    /// Resize the canvas and reset state to force a new baseline.
    #[allow(dead_code)]
    pub fn resize_backbuffer(&mut self, dims: FrameDimensions) {
        let mut s = self.shared.lock();
        let ts = choose_tile_size(s.width, s.height);
        s.width = dims.width.max(1);
        s.height = dims.height.max(1);
        s.pixel_backbuffer_rgba = vec![0; s.width * s.height * 4];
        s.current_total_squared_error = None;
        s.best_total_squared_error = None;
        s.stats = EvolutionStats { max_triangles_cap: s.max_triangles_cap, ..Default::default() };
        s.generation_counter = 0;
        s.tile_grid = Some(TileGrid::new(s.width, s.height, ts));
        s.spatial_index = Some(TriangleSpatialIndex::new(s.width, s.height, ts));
        s.dna.triangles.clear();
    }

    /// Provide a target image to approximate. If size differs, the canvas is resized.
    pub fn set_target_image(&mut self, rgba: Vec<u8>, width: usize, height: usize) {
        let mut s = self.shared.lock();
        let ts = choose_tile_size(s.width, s.height);
        if s.width != width || s.height != height {
            s.width = width.max(1);
            s.height = height.max(1);
            s.pixel_backbuffer_rgba = vec![0; s.width * s.height * 4];
            s.tile_grid = Some(TileGrid::new(s.width, s.height, ts));
            s.spatial_index = Some(TriangleSpatialIndex::new(s.width, s.height, ts));
        }
        s.target_image_rgba = Some(rgba);

        // force baseline rebuild on next run
        s.current_total_squared_error = None;
        s.best_total_squared_error = None;
        s.stats = EvolutionStats { max_triangles_cap: s.max_triangles_cap, ..Default::default() };
        s.generation_counter = 0;
        s.dna.triangles.clear();
    }

    /// Copy the current canvas for UI upload.
    pub fn capture_snapshot(&self) -> (Vec<u8>, usize, usize, u64) {
        let s = self.shared.lock();
        (s.pixel_backbuffer_rgba.clone(), s.width, s.height, s.generation_counter)
    }

    /// Get a copy of current stats for the HUD.
    pub fn capture_stats_snapshot(&self) -> EvolutionStats {
        let s = self.shared.lock();
        let mut stats = s.stats.clone();
        stats.max_triangles_cap = s.max_triangles_cap;
        stats
    }
}
