// -----------------------------------------------------------------------------
// Stats exposed to the UI
// -----------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct EvolutionStats {
    pub recent_acceptance_percent: f32,
    pub recent_window_size: usize,

    pub proposals_per_second: f32,
    pub accepts_per_second: f32,

    pub current_error: Option<u64>,
    pub best_error: Option<u64>,
    pub last_accept_delta: Option<i64>,

    pub last_operator_label: Option<String>,
    pub last_tiles_touched: Option<usize>,

    pub operator_weights: Vec<(String, f32)>,

    pub triangle_count: usize,
    pub generation_counter: u64,

    pub total_proposals: u64,
    pub total_accepts: u64,

    pub error_history: Vec<u64>,

    pub max_triangles_cap: usize,

    pub anneal_temp: Option<f64>,

    // Uphill acceptance (windowed + totals)
    pub recent_uphill_window_size: usize,      // uphill attempts seen in the last HUD window
    pub recent_uphill_acceptance_percent: f32, // uphill accepts / uphill attempts * 100 over last window
    pub total_uphill_attempts: u64,            // lifetime uphill attempts
    pub total_uphill_accepts: u64,             // lifetime uphill accepts
}

impl EvolutionStats {
    pub fn push_best_error_history(&mut self, v: u64) {
        const MAX: usize = 512;
        self.error_history.push(v);
        if self.error_history.len() > MAX {
            let extra = self.error_history.len() - MAX;
            self.error_history.drain(0..extra);
        }
    }
}