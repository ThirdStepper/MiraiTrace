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
use rayon::prelude::*;

use super::{
    fill_region_rgba_solid, compose_candidate_region_with_culled_subset,
    total_squared_error_rgb_region_from_buffer_vs_target,
    total_squared_error_rgb_region_from_canvas_vs_target,
    total_sse_region_with_cutoff, pad_and_clamp_rect,
    AdaptiveMutationScheduler, MutationOperator,
    FrameDimensions, IntRect, Triangle,
    TileGrid, TriangleSpatialIndex, choose_tile_size,
    TriangleDna, SimAnneal,
};
use super::mutation::{propose_mutation_with_bbox, generate_micro_variations};
use super::EvolutionStats;
use super::compute_backend::{ComputeBackend, CpuBackend, WgpuBackend, ProposalEvaluationData};

pub use crate::engine::ComputeBackendType;

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

    // compute backend selection
    compute_backend: ComputeBackendType,
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
            compute_backend: ComputeBackendType::Cpu,
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

            // Initialize compute backend based on user selection
            let mut backend: Box<dyn ComputeBackend> = {
                let s = shared_mutex.lock();
                let (backend_type, w, h) = (s.compute_backend, s.width, s.height);
                drop(s); // Release lock before potentially slow GPU init

                match backend_type {
                    ComputeBackendType::Cpu => Box::new(CpuBackend::new()),
                    ComputeBackendType::Wgpu => {
                        let mut gpu_backend = WgpuBackend::new();
                        if let Err(e) = gpu_backend.initialize(w, h) {
                            eprintln!("Failed to initialize WGPU backend: {}, falling back to CPU", e);
                            Box::new(CpuBackend::new())
                        } else {
                            Box::new(gpu_backend)
                        }
                    }
                }
            };

            // Track canvas dimensions and backend type to detect when we need to reinitialize
            let mut last_canvas_w = 0;
            let mut last_canvas_h = 0;
            let mut last_backend_type = { let s = shared_mutex.lock(); s.compute_backend };

            // Track actual backend name for stats display
            let mut actual_backend_name = String::from("CPU");

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

                // Re-initialize backend if canvas size or backend type changed
                let current_backend_type = { let s = shared_mutex.lock(); s.compute_backend };
                if canvas_w != last_canvas_w || canvas_h != last_canvas_h || current_backend_type != last_backend_type {
                    last_canvas_w = canvas_w;
                    last_canvas_h = canvas_h;
                    last_backend_type = current_backend_type;

                    // Reinitialize backend with new dimensions or type
                    backend = match current_backend_type {
                        ComputeBackendType::Cpu => {
                            actual_backend_name = String::from("CPU");
                            Box::new(CpuBackend::new())
                        },
                        ComputeBackendType::Wgpu => {
                            let mut gpu_backend = WgpuBackend::new();
                            if let Err(e) = gpu_backend.initialize(canvas_w, canvas_h) {
                                eprintln!("Failed to initialize WGPU backend: {}, falling back to CPU", e);
                                actual_backend_name = String::from("CPU (WGPU init failed)");
                                Box::new(CpuBackend::new())
                            } else {
                                actual_backend_name = String::from("WGPU GPU");
                                Box::new(gpu_backend)
                            }
                        }
                    };

                    // Update stats with actual backend
                    {
                        let mut s = shared_mutex.lock();
                        s.stats.active_backend = actual_backend_name.clone();
                    }
                }

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
                    // Check pause flag frequently (every iteration, not just outer loop)
                    if !running_flag.load(Ordering::Relaxed) {
                        break;
                    }

                    // Snapshot live state needed to score a proposal
                    let (total_err_opt, dna_snapshot, max_tris_cap, grid_snapshot, index_snapshot) = {
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

                    // Choose an operator
                    let op = scheduler.sample_operator(&mut rng, dna_snapshot.triangles.len(), growth_target);

                    // BATCH MUTATION SAMPLING: Try N candidates, keep the best
                    // Scale batch size by resolution to avoid excessive GPU/CPU work
                    let pixels = canvas_w * canvas_h;
                    let base_batch_size = if pixels > 800 * 800 {
                        5  // High res: smaller batches
                    } else if pixels > 400 * 400 {
                        10  // Medium res
                    } else {
                        15  // Low res: larger batches
                    };

                    let batch_size = if dna_snapshot.triangles.len() < growth_target {
                        base_batch_size + 5  // Slightly larger during growth
                    } else {
                        base_batch_size  // Base size during refinement
                    };

                    // BATCH SAMPLING: Generate all proposals first (sequential with shared RNG)
                    let mut proposals = Vec::with_capacity(batch_size);
                    for _ in 0..batch_size {
                        proposals.push(propose_mutation_with_bbox(
                            &dna_snapshot, &mut rng,
                            canvas_w as i32, canvas_h as i32,
                            max_tris_cap, op,
                            &grid_snapshot, &target_rgba,
                        ));
                    }

                    // Get canvas snapshot for batch evaluation
                    let canvas_snapshot = {
                        let s = shared_mutex.lock();
                        s.pixel_backbuffer_rgba.clone()
                    };

                    // GPU BATCH EVALUATION: Convert proposals to GPU format with spatial culling
                    // This matches the CPU path logic for correct triangle rendering
                    let gpu_proposals: Vec<ProposalEvaluationData> = proposals
                        .iter()
                        .map(|p| {
                            let union_rect = pad_and_clamp_rect(p.affected_bbox_px, 2, canvas_w, canvas_h);
                            if union_rect.w == 0 || union_rect.h == 0 { return None; }

                            // Clone spatial index for mutation (needed by triangles_overlapping_region)
                            let mut index_local = index_snapshot.clone();

                            // Use spatial index to find triangles overlapping this region (same as CPU path)
                            let mut overlapping_indices = index_local.triangles_overlapping_region(
                                &union_rect, canvas_w, canvas_h, dna_snapshot.triangles.len()
                            );

                            // Ensure the changed triangle is included (it might have moved into region)
                            if let Some(ci) = p.changed_index {
                                if !overlapping_indices.iter().any(|&v| v == ci) {
                                    overlapping_indices.push(ci);
                                    overlapping_indices.sort_unstable();
                                }
                            }

                            // Build triangle list: candidate for changed index, current DNA for others
                            let triangles_for_gpu: Vec<Triangle> = overlapping_indices.iter().map(|&idx| {
                                if Some(idx) == p.changed_index {
                                    p.candidate_dna_out.triangles[idx].clone()
                                } else {
                                    dna_snapshot.triangles[idx].clone()
                                }
                            }).collect();

                            Some(ProposalEvaluationData {
                                region: union_rect,
                                triangles: triangles_for_gpu,
                                triangle_indices: (0..overlapping_indices.len()).collect(),
                            })
                        })
                        .filter_map(|x| x)
                        .collect();

                    // Early exit if no valid proposals
                    if gpu_proposals.is_empty() {
                        sa.note_no_improvement();
                        continue;
                    }

                    // Call GPU batch evaluation (or CPU fallback)
                    let region_sse_results = backend.evaluate_proposals_batch(
                        &canvas_snapshot,
                        &target_rgba,
                        canvas_w,
                        canvas_h,
                        bg_rgba,
                        &gpu_proposals,
                    );

                    // Handle GPU errors by falling back to CPU
                    let region_sse_results = match region_sse_results {
                        Ok(results) => results,
                        Err(e) => {
                            eprintln!("GPU batch evaluation failed: {}, falling back to CPU", e);

                            // Update stats to show we're using CPU fallback
                            if actual_backend_name.starts_with("WGPU") {
                                actual_backend_name = String::from("CPU (GPU error fallback)");
                                let mut s = shared_mutex.lock();
                                s.stats.active_backend = actual_backend_name.clone();
                            }
                            // Evaluate using parallel CPU path as fallback
                            proposals
                                .par_iter()
                                .filter_map(|proposal| {
                                    let mut index_snapshot_local = index_snapshot.clone();
                                    let mut scratch_region_local = Vec::new();

                                    let union_rect = pad_and_clamp_rect(proposal.affected_bbox_px, 2, canvas_w, canvas_h);
                                    if union_rect.w == 0 || union_rect.h == 0 { return None; }

                                    let sse_rect = pad_and_clamp_rect(union_rect, 2, canvas_w, canvas_h);
                                    let tiles_touched = grid_snapshot.tiles_overlapping_rect(&sse_rect, canvas_w, canvas_h);
                                    if tiles_touched.is_empty() { return None; }

                                    scratch_region_local.resize(union_rect.w * union_rect.h * 4, 0);
                                    fill_region_rgba_solid(&mut scratch_region_local, &union_rect, bg_rgba);
                                    compose_candidate_region_with_culled_subset(
                                        &mut scratch_region_local, &union_rect, canvas_w, canvas_h,
                                        &dna_snapshot, &mut index_snapshot_local,
                                        proposal.changed_index, &proposal.candidate_dna_out,
                                    );

                                    let current_error_in_rect = total_squared_error_rgb_region_from_canvas_vs_target(
                                        &canvas_snapshot, &target_rgba, canvas_w, &union_rect);

                                    let candidate_error_in_rect = total_sse_region_with_cutoff(
                                        &scratch_region_local, &target_rgba, canvas_w, &union_rect, current_error_in_rect);

                                    Some(candidate_error_in_rect)
                                })
                                .collect()
                        }
                    };

                    // Sort proposals by SSE (best to worst)
                    // Try up to top 3 proposals to balance exploration vs overhead
                    let mut sorted_proposals: Vec<(usize, u64)> = region_sse_results
                        .iter()
                        .enumerate()
                        .map(|(i, &sse)| (i, sse))
                        .collect();

                    if sorted_proposals.is_empty() {
                        sa.note_no_improvement();
                        continue;
                    }

                    sorted_proposals.sort_by_key(|(_, sse)| *sse);

                    // Try proposals from best to worst until one is accepted (limit to top 3)
                    let mut accepted_proposal: Option<usize> = None;
                    let max_tries = 3.min(sorted_proposals.len());

                    for &(proposal_idx, _) in sorted_proposals.iter().take(max_tries) {
                        let proposal = &proposals[proposal_idx];
                        let union_rect = pad_and_clamp_rect(proposal.affected_bbox_px, 2, canvas_w, canvas_h);
                        let sse_rect = pad_and_clamp_rect(union_rect, 2, canvas_w, canvas_h);
                        let _tiles_touched = grid_snapshot.tiles_overlapping_rect(&sse_rect, canvas_w, canvas_h);

                        // Re-render the candidate to get the scratch_region buffer
                        // (GPU evaluation doesn't return the actual pixels, only SSE)
                        scratch_region.resize(union_rect.w * union_rect.h * 4, 0);
                        fill_region_rgba_solid(&mut scratch_region, &union_rect, bg_rgba);
                        let mut index_snapshot_local = index_snapshot.clone();
                        compose_candidate_region_with_culled_subset(
                            &mut scratch_region, &union_rect, canvas_w, canvas_h,
                            &dna_snapshot, &mut index_snapshot_local,
                            proposal.changed_index, &proposal.candidate_dna_out,
                        );

                        // Final exact recompute (no cutoff) before acceptance decision
                        let exact_current_in_rect = {
                            let s = shared_mutex.lock();
                            total_squared_error_rgb_region_from_canvas_vs_target(&s.pixel_backbuffer_rgba, &target_rgba, canvas_w, &union_rect)
                        };

                        let exact_candidate_in_rect = total_squared_error_rgb_region_from_buffer_vs_target(
                            &scratch_region, &target_rgba, canvas_w, &union_rect);

                        let candidate_total_error = match current_total_error.checked_sub(exact_current_in_rect) {
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

                        let accepted = sa.should_accept_area(&mut rng, delta, region_area, exact_current_in_rect);
                        let is_uphill = delta > 0;

                        // Update HUD counters for this attempt
                        {
                            let mut s = shared_mutex.lock();
                            s.proposals_counter_window = s.proposals_counter_window.saturating_add(1);
                            s.stats.total_proposals = s.stats.total_proposals.saturating_add(1);

                            if is_uphill {
                                s.uphill_attempts_counter_window = s.uphill_attempts_counter_window.saturating_add(1);
                                s.stats.total_uphill_attempts = s.stats.total_uphill_attempts.saturating_add(1);
                                if accepted {
                                    s.uphill_accepts_counter_window = s.uphill_accepts_counter_window.saturating_add(1);
                                    s.stats.total_uphill_accepts = s.stats.total_uphill_accepts.saturating_add(1);
                                }
                            }
                        }

                        if accepted {
                            // Store the accepted proposal index and break out
                            accepted_proposal = Some(proposal_idx);
                            break;
                        }
                        // If not accepted, try next proposal in sorted order
                    }

                    // After trying all proposals, check if any was accepted
                    let (_proposal_idx, proposal) = match accepted_proposal {
                        Some(idx) => (idx, &proposals[idx]),
                        None => {
                            // None of the proposals were accepted
                            sa.note_no_improvement();
                            continue;
                        }
                    };

                    // Re-extract the winning proposal data (we need this for commit)
                    let union_rect = pad_and_clamp_rect(proposal.affected_bbox_px, 2, canvas_w, canvas_h);
                    let sse_rect = pad_and_clamp_rect(union_rect, 2, canvas_w, canvas_h);
                    let tiles_touched = grid_snapshot.tiles_overlapping_rect(&sse_rect, canvas_w, canvas_h);

                    // Re-render accepted proposal for commit
                    scratch_region.resize(union_rect.w * union_rect.h * 4, 0);
                    fill_region_rgba_solid(&mut scratch_region, &union_rect, bg_rgba);
                    let mut index_snapshot_local = index_snapshot.clone();
                    compose_candidate_region_with_culled_subset(
                        &mut scratch_region, &union_rect, canvas_w, canvas_h,
                        &dna_snapshot, &mut index_snapshot_local,
                        proposal.changed_index, &proposal.candidate_dna_out,
                    );

                    // Final recompute for the accepted proposal
                    let exact_current_in_rect = {
                        let s = shared_mutex.lock();
                        total_squared_error_rgb_region_from_canvas_vs_target(&s.pixel_backbuffer_rgba, &target_rgba, canvas_w, &union_rect)
                    };

                    let exact_candidate_in_rect = total_squared_error_rgb_region_from_buffer_vs_target(
                        &scratch_region, &target_rgba, canvas_w, &union_rect);

                    let candidate_total_error = match current_total_error.checked_sub(exact_current_in_rect) {
                        Some(rest) => rest + exact_candidate_in_rect,
                        None => {
                            let base = grid_snapshot.sse_per_tile.iter().copied().sum::<u64>().max(current_total_error);
                            base.saturating_sub(exact_current_in_rect) + exact_candidate_in_rect
                        }
                    };

                    let delta = candidate_total_error as i64 - current_total_error as i64;
                    let _is_uphill = delta > 0;

                    // Update scheduler bias (accepted is always true here since we found one)
                    scheduler.record_outcome(op, true);
                    let op_weights = scheduler.current_weights();

                    // Update HUD counters and stats
                    {
                        let mut s = shared_mutex.lock();

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
                            ("Remove".into(), op_weights[1]),
                            ("Move".into(), op_weights[2]),
                            ("Recolor".into(), op_weights[3]),
                        ];
                        s.stats.max_triangles_cap = s.max_triangles_cap;

                        // keep exporting annealer temp (so HUD can show it)
                        s.stats.anneal_temp = Some(sa.temp());
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
                            MutationOperator::RemoveTriangle => {
                                // Removal shifts all indices down - just rebuild
                                let dna_clone = s.dna.clone();
                                si.rebuild(&dna_clone, w2, h2);
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

                    // ========== MICRO-OPTIMIZATION (Evolve-style local refinement) ==========
                    // After accepting a mutation, try small variations to "snap" to a better local optimum
                    // This mimics tux3/Evolve's aggressive post-acceptance optimization
                    if let Some(changed_idx) = proposal.changed_index {
                        // Get current DNA snapshot
                        let micro_dna = { shared_mutex.lock().dna.clone() };

                        // Generate 8 micro-variations (color, alpha, vertex tweaks)
                        let micro_proposals = generate_micro_variations(
                            &micro_dna,
                            changed_idx,
                            canvas_w as i32,
                            canvas_h as i32,
                            &mut rng,
                            8,
                        );

                        if !micro_proposals.is_empty() {
                            // Convert to GPU format with spatial culling (same as main batch)
                            let micro_gpu_proposals: Vec<ProposalEvaluationData> = micro_proposals
                                .iter()
                                .map(|p| {
                                    let union_rect = pad_and_clamp_rect(p.affected_bbox_px, 2, canvas_w, canvas_h);
                                    if union_rect.w == 0 || union_rect.h == 0 { return None; }

                                    let mut index_local = index_snapshot.clone();
                                    let mut overlapping_indices = index_local.triangles_overlapping_region(
                                        &union_rect, canvas_w, canvas_h, micro_dna.triangles.len()
                                    );

                                    if let Some(ci) = p.changed_index {
                                        if !overlapping_indices.iter().any(|&v| v == ci) {
                                            overlapping_indices.push(ci);
                                            overlapping_indices.sort_unstable();
                                        }
                                    }

                                    let triangles_for_gpu: Vec<Triangle> = overlapping_indices.iter().map(|&idx| {
                                        if Some(idx) == p.changed_index {
                                            p.candidate_dna_out.triangles[idx].clone()
                                        } else {
                                            micro_dna.triangles[idx].clone()
                                        }
                                    }).collect();

                                    Some(ProposalEvaluationData {
                                        region: union_rect,
                                        triangles: triangles_for_gpu,
                                        triangle_indices: (0..overlapping_indices.len()).collect(),
                                    })
                                })
                                .filter_map(|x| x)
                                .collect();

                            // Evaluate all micro-variations in parallel (GPU or CPU)
                            if let Ok(micro_sse_results) = backend.evaluate_proposals_batch(
                                &canvas_snapshot, &target_rgba, canvas_w, canvas_h, bg_rgba, &micro_gpu_proposals
                            ) {
                                // Find best micro-variation (pure hill climbing for micro-opts)
                                if let Some((best_idx, &best_sse)) = micro_sse_results.iter().enumerate().min_by_key(|(_, &sse)| sse) {
                                    let current_sse = {
                                        let s = shared_mutex.lock();
                                        s.current_total_squared_error.unwrap_or(u64::MAX)
                                    };

                                    // Only accept if micro-variation improves (greedy for micro-opts)
                                    if best_sse < current_sse {
                                        let best_micro = &micro_proposals[best_idx];
                                        let micro_rect = pad_and_clamp_rect(best_micro.affected_bbox_px, 2, canvas_w, canvas_h);

                                        // Re-render best micro-variation for commit
                                        scratch_region.resize(micro_rect.w * micro_rect.h * 4, 0);
                                        fill_region_rgba_solid(&mut scratch_region, &micro_rect, bg_rgba);
                                        let mut index_local = index_snapshot.clone();
                                        compose_candidate_region_with_culled_subset(
                                            &mut scratch_region, &micro_rect, canvas_w, canvas_h,
                                            &micro_dna, &mut index_local,
                                            best_micro.changed_index, &best_micro.candidate_dna_out,
                                        );

                                        // Commit best micro-variation (simplified - no SA, direct commit)
                                        let mut s = shared_mutex.lock();
                                        let ts = choose_tile_size(s.width, s.height);
                                        let mut tg = s.tile_grid.take().unwrap_or_else(|| TileGrid::new(s.width, s.height, ts));

                                        tg.blit_region_and_update_sse(
                                            &mut s.pixel_backbuffer_rgba,
                                            &target_rgba,
                                            canvas_w,
                                            &micro_rect,
                                            &scratch_region,
                                        );

                                        let tiles_sum: u64 = tg.sse_per_tile.iter().copied().sum();
                                        s.current_total_squared_error = Some(tiles_sum);
                                        s.tile_grid = Some(tg);
                                        s.dna = best_micro.candidate_dna_out.clone();

                                        if s.best_total_squared_error.map(|b| tiles_sum < b).unwrap_or(true) {
                                            s.best_total_squared_error = Some(tiles_sum);
                                            s.stats.best_error = Some(tiles_sum);
                                            s.stats.push_best_error_history(tiles_sum);
                                        }

                                        // Update spatial index for micro-variation (simplified - just update the changed triangle)
                                        let (w2, h2) = (s.width, s.height);
                                        let mut si = s.spatial_index.take().unwrap_or_else(|| TriangleSpatialIndex::new(w2, h2, ts));
                                        if let Some(idx) = best_micro.changed_index {
                                            if let Some(ref old_tri) = best_micro.old_triangle_for_update {
                                                si.remove_triangle(idx, old_tri, w2, h2);
                                            }
                                            let tri_now = &s.dna.triangles[idx];
                                            si.insert_triangle(idx, tri_now, w2, h2);
                                        }
                                        s.spatial_index = Some(si);
                                    }
                                }
                            }
                        }
                    }
                    // ========== END MICRO-OPTIMIZATION ==========

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

            // Scale work budget by resolution to avoid excessive work at high resolutions
            let pixels = s.width * s.height;
            s.work_budget_per_tick = if pixels > 800 * 800 {
                250  // High res: 250 iterations per tick (still 5K-10K proposals evaluated)
            } else if pixels > 400 * 400 {
                500  // Medium res
            } else {
                1000  // Low res (reduced from 5000 - was excessive)
            };
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

    /// Save current DNA to JSON file
    pub fn save_dna_to_file(&self, path: &std::path::Path) -> Result<(), String> {
        let s = self.shared.lock();
        let json = serde_json::to_string_pretty(&s.dna)
            .map_err(|e| format!("Failed to serialize DNA: {}", e))?;
        std::fs::write(path, json)
            .map_err(|e| format!("Failed to write file: {}", e))?;
        Ok(())
    }

    /// Load DNA from JSON file and replace current DNA
    pub fn load_dna_from_file(&mut self, path: &std::path::Path) -> Result<(), String> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        let dna: TriangleDna = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to deserialize DNA: {}", e))?;

        // Replace DNA and force rebaseline
        let mut s = self.shared.lock();
        let ts = choose_tile_size(s.width, s.height);
        s.dna = dna;
        s.current_total_squared_error = None;  // Force rebaseline
        s.best_total_squared_error = None;
        s.generation_counter = 0;
        s.stats.triangle_count = s.dna.triangles.len();

        // Rebuild spatial index for new DNA
        let (w, h) = (s.width, s.height);
        let mut si = TriangleSpatialIndex::new(w, h, ts);
        si.rebuild(&s.dna, w, h);
        s.spatial_index = Some(si);

        Ok(())
    }

    /// Export current DNA to SVG file
    pub fn export_svg(&self, path: &std::path::Path) -> Result<(), String> {
        let s = self.shared.lock();
        let width = s.width;
        let height = s.height;
        let triangles = &s.dna.triangles;

        let mut svg = format!(
            "<svg width=\"{}\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
            width, height
        );

        // Add background rectangle
        let bg = s.background_color_rgba;
        svg.push_str(&format!(
            "  <rect width=\"100%\" height=\"100%\" fill=\"rgb({},{},{})\" />\n",
            bg[0], bg[1], bg[2]
        ));

        // Add all triangles
        for tri in triangles {
            svg.push_str(&format!(
                "  <polygon points=\"{},{} {},{} {},{}\" fill=\"rgb({},{},{})\" opacity=\"{}\" />\n",
                tri.x0, tri.y0,
                tri.x1, tri.y1,
                tri.x2, tri.y2,
                tri.r, tri.g, tri.b,
                tri.a as f32 / 255.0
            ));
        }

        svg.push_str("</svg>");

        std::fs::write(path, svg)
            .map_err(|e| format!("Failed to write SVG: {}", e))?;

        Ok(())
    }

    /// Get the currently selected compute backend
    pub fn get_compute_backend(&self) -> ComputeBackendType {
        let s = self.shared.lock();
        s.compute_backend
    }

    /// Set the compute backend (CPU vs OpenCL GPU)
    pub fn set_compute_backend(&mut self, backend: ComputeBackendType) {
        let mut s = self.shared.lock();
        s.compute_backend = backend;
        // Note: Actual backend implementation in parallel evaluation would go here
        // For now, this just tracks the selection
    }
}
