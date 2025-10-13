// -----------------------------------------------------------------------------
// Simple acceptance policy: Greedy hill climbing with occasional random jumps
// -----------------------------------------------------------------------------

use rand::Rng;

/// Much simpler acceptance strategy:
/// - Always accept improvements (greedy)
/// - Occasionally accept random changes to escape local minima
pub struct SimAnneal {
    /// Generations since last improvement
    no_improve_iters: u32,
    /// After this many generations without improvement, force accept next mutation
    force_accept_threshold: u32,
    /// Probability of accepting a random mutation when stuck
    random_jump_probability: f64,
}

impl SimAnneal {
    pub fn new() -> Self {
        Self {
            no_improve_iters: 0,
            force_accept_threshold: 2000,  // Force progress after 2000 stuck iterations
            random_jump_probability: 0.02,  // 2% chance of random jump when stuck
        }
    }

    /// Simple greedy acceptance: accept if it improves, or if we're stuck
    pub fn should_accept_area<R: Rng>(
        &mut self,
        rng: &mut R,
        delta_sse: i64,
        _region_area: usize,
        _current_error_in_rect: u64,
    ) -> bool {
        // Always accept improvements
        if delta_sse <= 0 {
            return true;
        }

        // If we're really stuck, force accept occasionally to explore
        if self.no_improve_iters > self.force_accept_threshold {
            if rng.gen::<f64>() < self.random_jump_probability {
                return true;
            }
        }

        // Otherwise reject bad mutations
        false
    }

    /// No-op for compatibility
    pub fn set_phase(&mut self, _pre_cap: bool) {
        // Phase doesn't matter in greedy approach
    }

    /// Track whether we improved
    pub fn note_improvement(&mut self) {
        self.no_improve_iters = 0;
    }

    pub fn note_no_improvement(&mut self) {
        self.no_improve_iters = self.no_improve_iters.saturating_add(1);
    }

    /// Return a fake temperature for UI display
    pub fn temp(&self) -> f64 {
        if self.no_improve_iters > self.force_accept_threshold {
            0.02 // Show "hot" when stuck
        } else {
            0.0 // Show "cold" when improving
        }
    }
}
