// engine/mod.rs
mod engine_core;
mod types;
mod geom;
mod triangle;
mod tile_grid;
mod spatial_index;
mod raster;
mod mutation;
mod scheduler;
mod annealer;
mod stats;
mod compute_backend;

use compute_backend::ComputeBackend;

pub use engine_core::EvolutionEngine;
pub use types::FrameDimensions;
pub use stats::EvolutionStats;
pub use mutation::MutationOperator;
pub use annealer::SimAnneal;

/// Compute backend selection
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeBackendType {
    Cpu,
    Wgpu,
}

impl ComputeBackendType {
    pub fn is_available(&self) -> bool {
        match self {
            ComputeBackendType::Cpu => true,
            ComputeBackendType::Wgpu => {
                compute_backend::WgpuBackend::is_available()
            }
        }
    }

    pub fn name(&self) -> &str {
        match self {
            ComputeBackendType::Cpu => "CPU",
            ComputeBackendType::Wgpu => "WGPU GPU",
        }
    }
}


// Make IntRect available to submodules
pub(crate) use types::IntRect;
pub(crate) use geom::{clamp_i32, union_rect, pad_and_clamp_rect};
pub(crate) use triangle::{Triangle, TriangleDna, triangle_bbox_px};
pub(crate) use tile_grid::{TileGrid, choose_tile_size};
pub(crate) use spatial_index::{TriangleSpatialIndex};
pub(crate) use scheduler::AdaptiveMutationScheduler;
pub(crate) use raster::{
    fill_region_rgba_solid,
    total_squared_error_rgb_region_from_buffer_vs_target,
    total_squared_error_rgb_region_from_canvas_vs_target,
    total_sse_region_with_cutoff,
    compose_candidate_region_with_culled_subset,
};