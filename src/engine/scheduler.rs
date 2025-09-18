// -----------------------------------------------------------------------------
// Adaptive scheduler
// -----------------------------------------------------------------------------
use rand::prelude::*;
use rand_pcg::Pcg64Mcg as PcgRng;

use super::MutationOperator;

/// Adaptive scheduler with an early **growth bias** to prefer AddTriangle
/// until we reach a target triangle count.
pub(crate) struct AdaptiveMutationScheduler {
    base_weights: [f32; MutationOperator::COUNT],
    weights: [f32; MutationOperator::COUNT],
    proposals: [u64; MutationOperator::COUNT],
    accepts: [u64; MutationOperator::COUNT],
    accept_boost: f32,
    decay_toward_base: f32,
}

impl AdaptiveMutationScheduler {
    pub(crate) fn new() -> Self {
        // Heavier bias on Add: we want to "lay down paint" before refinement.
        let base = [0.33, 0.33, 0.33]; // Add, Move, Recolor
        Self {
            base_weights: base,
            weights: base,
            proposals: [0; MutationOperator::COUNT],
            accepts: [0; MutationOperator::COUNT],
            accept_boost: 0.20,
            decay_toward_base: 0.05,
        }
    }

    /// Sample an operator with growth bias while triangle count < growth_target.
    pub(crate) fn sample_operator(
        &mut self,
        rng: &mut PcgRng,
        current_triangle_count: usize,
        growth_target: usize,
    ) -> MutationOperator {
        let mut probs = self.weights;

        if current_triangle_count < growth_target {
            // How far we are from the target (0..1)
            let deficit = (growth_target - current_triangle_count) as f32 / growth_target.max(1) as f32;

            // Tunables:
            const GROWTH_STRENGTH: f32 = 0.45;  // was 0.85 → much gentler
            const GROWTH_GAMMA: f32    = 0.6;   // sub-linear curve (sqrt-like)
            const MAX_DAMP: f32        = 0.15;  // at most 20% reduction to Move/Recolor

            // Sub-linear tilt toward Add
            let tilt = (GROWTH_STRENGTH * deficit.powf(GROWTH_GAMMA)).clamp(0.0, GROWTH_STRENGTH);

            // Add gets a little extra mass
            probs[MutationOperator::AddTriangle.index()] += tilt;

            // Move/Recolor get a gentle, deficit-shaped reduction (not a hard 0.9)
            let damp = 1.0 - MAX_DAMP * deficit.powf(GROWTH_GAMMA);
            probs[MutationOperator::MoveVertex.index()]   *= damp;
            probs[MutationOperator::Recolor.index()]      *= damp;
        }

        // Normalize
        let sum: f32 = probs.iter().map(|v| v.max(1e-6)).sum();
        for v in &mut probs {
            *v = (*v).max(1e-6) / sum;
        }

        // Multinomial draw
        let r = rng.gen::<f32>();
        let mut acc = 0.0;
        for (i, &w) in probs.iter().enumerate() {
            acc += w;
            if r <= acc {
                return match i {
                    0 => MutationOperator::AddTriangle,
                    1 => MutationOperator::MoveVertex,
                    _ => MutationOperator::Recolor,
                };
            }
        }
        MutationOperator::Recolor
    }

    pub(crate) fn record_outcome(&mut self, op: MutationOperator, accepted: bool) {
        let i = op.index();
        self.proposals[i] = self.proposals[i].saturating_add(1);

        // Gentle decay toward base for all operators
        for k in 0..MutationOperator::COUNT {
            self.weights[k] = self.weights[k] * (1.0 - self.decay_toward_base)
                + self.base_weights[k] * self.decay_toward_base;
        }

        if accepted {
            self.accepts[i] = self.accepts[i].saturating_add(1);
            self.weights[i] += self.accept_boost;
        }

        // Normalize
        let sum: f32 = self.weights.iter().map(|v| v.max(1e-6)).sum();
        for v in &mut self.weights {
            *v = (*v).max(1e-6) / sum;
        }
    }

    pub(crate) fn current_weights(&self) -> [f32; MutationOperator::COUNT] {
        self.weights
    }
}