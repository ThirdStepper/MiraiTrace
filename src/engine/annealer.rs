// -----------------------------------------------------------------------------
// Simulated annealing helper
// -----------------------------------------------------------------------------

use rand::Rng;


pub struct SimAnneal {
    temp: f64,
    decay: f64,
    min_temp: f64,
    reheat_mult: f64,

    window: u32,
    tried_uphill: u32,
    accepted_uphill: u32,

    target_up_low: f64,
    target_up_high: f64,

    no_improve_iters: u32,
    plateau_limit: u32,

    cold_guard_temp: f64,   
    cold_guard_dcap: f64,   

    pub ppsse_ewma: f64,   // should refactor into being private later
    ppsse_alpha: f64,  
}

impl SimAnneal {
    pub fn new() -> Self {
        Self {
            temp: 0.20,             // start cooler now that normalization is “tighter”
            decay: 0.9995,          // slow, steady cooling
            min_temp: 0.001,         // can get quite greedy later
            reheat_mult: 1.06,      // mild reheats

            // acceptance tracking (UPHILL ONLY)
            window: 1024,
            tried_uphill: 0,
            accepted_uphill: 0,

            // target uphill acceptance band
            target_up_low: 0.01,
            target_up_high: 0.15,

            // plateau: no improvements (global) for a while
            no_improve_iters: 0,
            plateau_limit: 14_000,

            // safety clamps
            cold_guard_temp: 0.02,  // below this, be stricter on uphill
            cold_guard_dcap: 0.03,  // if normalized d > this, disallow uphill when cold

            // Rolling estimate of "SSE per pixel" used to scale region_area
            ppsse_ewma: 0.0,        // running per-pixel SSE estimate 
            ppsse_alpha: 0.04,      // EWMA smoothing factor, e.g., 0.05
        }
    }

    #[inline]
    fn update_ppsse_ewma(&mut self, current_error_in_rect: u64, region_area: usize) {
        if region_area == 0 { return; }
        let cur = (current_error_in_rect as f64) / (region_area as f64);
        if self.ppsse_ewma == 0.0 {
            self.ppsse_ewma = cur.max(1.0);
        } else {
            self.ppsse_ewma = (1.0 - self.ppsse_alpha) * self.ppsse_ewma + self.ppsse_alpha * cur.max(1.0);
        }
    }

    /// Tighten/relax uphill acceptance targets depending on phase.
    /// pre_cap = true while we're still growing triangle count toward the target.
    pub fn set_phase(&mut self, pre_cap: bool) {
        if pre_cap {
            // Early phase: be stricter on uphill so "spray" doesn't run away
            self.target_up_low  = 0.03;  // was 0.05
            self.target_up_high = 0.05;  // was 0.20
            self.cold_guard_dcap = 0.005; // block bigger uphill when cold
            // also gently cap temp if we've overheated
            if self.temp > 0.25 { self.temp = 0.25; }
        } else {
            // Post-cap: allow more exploratory uphill
            self.target_up_low  = 0.05;
            self.target_up_high = 0.20;
            self.cold_guard_dcap = 0.03;
        }
    }

    /// Optional helper if you want a manual cool nudge.
    pub fn nudge_cool(&mut self, factor: f64) {
        self.temp = (self.temp * factor).max(self.min_temp);
    }

    /// Area-normalized acceptance using an auto-calibrated per-pixel SSE scale.
    /// This preserves your "region_area" feel but keeps probabilities in range.
    pub fn should_accept_area<R: Rng>(
        &mut self,
        rng: &mut R,
        delta_sse: i64,
        region_area: usize,
        current_error_in_rect: u64, // used only to update EWMA scale
    ) -> bool {
        // Always accept downhill (not counted as uphill)
        if delta_sse <= 0 {
            // still update EWMA so scale tracks reality over time
            self.update_ppsse_ewma(current_error_in_rect, region_area);
            return true;
        }

        // Track an uphill attempt
        self.tried_uphill += 1;

        // Update rolling per-pixel SSE estimate
        self.update_ppsse_ewma(current_error_in_rect, region_area);

        // Effective normalization uses region_area * EWMA(per-pixel SSE)
        // Add a tiny epsilon so early frames don't overshoot.
        let scale = (region_area as f64) * (self.ppsse_ewma + 1.0);

        // Cold guard: when very cold, block big uphill moves
        let d = (delta_sse as f64) / scale.max(1.0);
        if self.temp <= self.cold_guard_temp && d > self.cold_guard_dcap {
            return false;
        }

        // Metropolis on the area-normalized delta
        let p = (-d / self.temp).exp();
        if rng.gen::<f64>() < p {
            self.accepted_uphill += 1;
            true
        } else {
            false
        }
    }

    /// Call once per *proposal attempt* (accepted or not).
    pub fn tick(&mut self) {
        // smooth cooling
        self.temp = (self.temp * self.decay).max(self.min_temp);

        // adapt every window using *uphill* acceptance rate
        if self.tried_uphill >= self.window {
            let acc = (self.accepted_uphill as f64) / (self.tried_uphill as f64 + 1e-9);
            if acc < self.target_up_low {
                // too stingy on uphill -> nudge temp up
                self.temp *= 1.07;
            } else if acc > self.target_up_high {
                // too generous on uphill -> cool faster a bit
                self.temp *= 0.95;
            }
            self.tried_uphill = 0;
            self.accepted_uphill = 0;
        }

        // plateau-based gentle reheat (regardless of uphill stats)
        if self.no_improve_iters >= self.plateau_limit {
            self.temp = (self.temp * self.reheat_mult).max(self.min_temp * 1.5);
            self.no_improve_iters = 0;
        }
    }

    pub fn note_improvement(&mut self) { self.no_improve_iters = 0; }
    pub fn note_no_improvement(&mut self) { self.no_improve_iters = self.no_improve_iters.saturating_add(1); }
    pub fn temp(&self) -> f64 { self.temp }
}