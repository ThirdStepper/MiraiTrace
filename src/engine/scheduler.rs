// -----------------------------------------------------------------------------
// Simple mutation scheduler:
// - Early: Always Add
// - Later: Equal mix of Remove/Move/Recolor
// -----------------------------------------------------------------------------
use rand::prelude::*;
use rand_pcg::Pcg64Mcg as PcgRng;

use super::MutationOperator;

/// Much simpler scheduler:
/// - While triangle count < target: Always AddTriangle
/// - Otherwise: equal probability for Remove/Move/Recolor
pub(crate) struct AdaptiveMutationScheduler {
    // Just for tracking/display
    proposals: [u64; MutationOperator::COUNT],
    accepts: [u64; MutationOperator::COUNT],
}

impl AdaptiveMutationScheduler {
    pub(crate) fn new() -> Self {
        Self {
            proposals: [0; MutationOperator::COUNT],
            accepts: [0; MutationOperator::COUNT],
        }
    }

    /// Simple rule-based sampling
    pub(crate) fn sample_operator(
        &mut self,
        rng: &mut PcgRng,
        current_triangle_count: usize,
        growth_target: usize,
    ) -> MutationOperator {
        // Phase 1: Build up triangles
        if current_triangle_count < growth_target {
            return MutationOperator::AddTriangle;
        }

        // Phase 2: Refine with equal probability
        let r = rng.gen_range(0..3);
        match r {
            0 => MutationOperator::RemoveTriangle,
            1 => MutationOperator::MoveVertex,
            _ => MutationOperator::Recolor,
        }
    }

    pub(crate) fn record_outcome(&mut self, op: MutationOperator, accepted: bool) {
        let i = op.index();
        self.proposals[i] = self.proposals[i].saturating_add(1);
        if accepted {
            self.accepts[i] = self.accepts[i].saturating_add(1);
        }
    }

    pub(crate) fn current_weights(&self) -> [f32; MutationOperator::COUNT] {
        // Return fake weights for UI display
        let mut weights = [0.0; MutationOperator::COUNT];
        for (i, w) in weights.iter_mut().enumerate() {
            if self.proposals[i] > 0 {
                *w = (self.accepts[i] as f32) / (self.proposals[i] as f32);
            }
        }
        // Normalize
        let sum: f32 = weights.iter().sum::<f32>().max(1e-6);
        for w in &mut weights {
            *w = *w / sum;
        }
        weights
    }
}
