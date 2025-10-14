//! UI layer
//! -----------------------
//! Top bar:
//!   • Open image…
//!   • Save canvas as PNG
//!   • Run / Pause
//!   • Canvas-only toggle
//!   • Canvas color picker  ← NEW (resets to a blank canvas with that color)
//!   • Max triangles slider (logarithmic, live)
//!   • HUD density selector (Auto / Compact / Full)
//!   • Mini-plot toggle (tiny sparkline of best error)
//!
//! Central panel:
//!   • Strict side-by-side Canvas | Original (aspect-fit in each half), or Canvas-only.
//!
//! Bottom bar (HUD):
//!   • Monospace, fixed-width columns to avoid jitter
//!   • Optional tiny sparkline for best-error history

use eframe::{egui, egui::ColorImage};
use egui::{Context, Margin, TextureHandle, TextureOptions, Vec2};
use std::path::Path;

use crate::engine::{EvolutionEngine, EvolutionStats, ComputeBackendType};

/// Main egui app.
pub struct MiraiTrace {
    // Core engine
    engine: EvolutionEngine,

    // Canvas texture (engine output) and the last generation uploaded to GPU.
    canvas_texture: Option<TextureHandle>,
    last_uploaded_generation: u64,

    // Original image texture and dimensions.
    original_texture: Option<TextureHandle>,
    original_width: usize,
    original_height: usize,

    // View options
    show_canvas_only: bool,

    // HUD options
    show_stats_window: bool,  // Toggle for floating stats window

    // Canvas clear color (UI model). Changing it clears/rebaselines the engine.
    canvas_clear_color: egui::Color32,
}

/// Fit an image of size (img_w, img_h) inside the available area while preserving aspect.
fn fit_image_inside(available: Vec2, img_w: usize, img_h: usize) -> Vec2 {
    if img_w == 0 || img_h == 0 {
        return Vec2::ZERO;
    }
    let aw = available.x.max(1.0);
    let ah = available.y.max(1.0);
    let iw = img_w as f32;
    let ih = img_h as f32;
    let scale = (aw / iw).min(ah / ih);
    Vec2::new((iw * scale).floor(), (ih * scale).floor())
}

impl MiraiTrace {
    /// Constructor used by your `main.rs`: takes an already-created engine.
    pub fn new(mut engine: EvolutionEngine) -> Self {
        // Keep HUD in sync with the current cap right away.
        engine.set_max_triangles(engine.max_triangles());

        Self {
            engine,
            canvas_texture: None,
            last_uploaded_generation: 0,
            original_texture: None,
            original_width: 0,
            original_height: 0,
            show_canvas_only: false,
            show_stats_window: true,  // Show stats by default
            canvas_clear_color: egui::Color32::from_rgba_unmultiplied(255, 255, 255, 255), // default white
        }
    }

    // -----------------------------
    // Image loading & saving
    // -----------------------------

    /// Decode an image from disk, set it as the engine target, and upload the “Original” texture.
    fn load_image_and_set_target(&mut self, ctx: &Context, path: &Path) -> Result<(), String> {
        // Decode via the `image` crate.
        let dyn_img = image::open(path).map_err(|e| format!("Failed to open image: {e}"))?;
        let rgba_img = dyn_img.to_rgba8();
        let (w, h) = rgba_img.dimensions();
        let (w_us, h_us) = (w as usize, h as usize);

        // Engine wants an owned Vec<u8>.
        let src_pixels: Vec<u8> = rgba_img.as_raw().clone();
        self.engine.set_target_image(src_pixels.clone(), w_us, h_us);

        // Upload an "Original" texture from the same pixels.
        let color_img = ColorImage::from_rgba_unmultiplied([w_us, h_us], &src_pixels);
        match &mut self.original_texture {
            Some(tex) => tex.set(color_img, TextureOptions::NEAREST),
            None => {
                let tex = ctx.load_texture("original-image", color_img, TextureOptions::NEAREST);
                self.original_texture = Some(tex);
            }
        }
        self.original_width = w_us;
        self.original_height = h_us;

        // Ensure the first new canvas frame after load is pushed to GPU.
        self.last_uploaded_generation = 0;
        Ok(())
    }

    /// Save the current canvas as a PNG (via Save dialog).
    fn save_canvas_as_png(&mut self) -> Result<(), String> {
        let Some(path) = rfd::FileDialog::new()
            .set_file_name("vectorgazo.png")
            .add_filter("PNG", &["png"])
            .save_file()
        else {
            return Ok(()); // user cancelled
        };

        let (pixels_rgba, w, h, _gen) = self.engine.capture_snapshot();
        if w == 0 || h == 0 || pixels_rgba.is_empty() {
            return Err("Canvas is empty".into());
        }

        image::save_buffer_with_format(
            &path,
            &pixels_rgba,
            w as u32,
            h as u32,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )
        .map_err(|e| format!("Failed to save PNG: {e}"))?;

        Ok(())
    }

    // -----------------------------
    // Textures & drawing helpers
    // -----------------------------

    /// Upload the engine canvas to a texture when the generation changes.
    fn maybe_upload_canvas(&mut self, ctx: &Context) {
        let (pixels_rgba, w, h, gen) = self.engine.capture_snapshot();
        if gen == self.last_uploaded_generation {
            return; // nothing new to upload this frame
        }

        let color_img = ColorImage::from_rgba_unmultiplied([w, h], &pixels_rgba);
        match &mut self.canvas_texture {
            Some(tex) => tex.set(color_img, TextureOptions::NEAREST),
            None => {
                let tex = ctx.load_texture("canvas", color_img, TextureOptions::NEAREST);
                self.canvas_texture = Some(tex);
            }
        }
        self.last_uploaded_generation = gen;
    }

    /// Draw one texture with aspect fit inside the current `ui`.
    fn draw_image_aspect_fit(ui: &mut egui::Ui, tex: &TextureHandle, img_w: usize, img_h: usize) {
        let available = ui.available_size();
        let size = fit_image_inside(available, img_w, img_h);
        if size != Vec2::ZERO {
            let image = egui::Image::new(tex).fit_to_exact_size(size);
            ui.add(image);
        } else {
            ui.label("No content");
        }
    }

    // -----------------------------
    // Bars (top & bottom)
    // -----------------------------

    /// Top bar with main controls, including **Canvas color** and **Max triangles**.
    fn top_bar(&mut self, ctx: &Context, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            // Open image…
            if ui
                .button("📂 Open image…")
                .on_hover_text("Choose a target image")
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "tga", "gif"])
                    .pick_file()
                {
                    let _ = self.load_image_and_set_target(ctx, &path);
                }
            }

            // Save canvas as PNG
            if ui
                .button("💾 Save PNG")
                .on_hover_text("Save the current canvas as a PNG")
                .clicked()
            {
                let _ = self.save_canvas_as_png();
            }

            // Export SVG
            if ui
                .button("📄 Export SVG")
                .on_hover_text("Export the current triangles as an SVG file")
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .set_file_name("vectorgazo.svg")
                    .add_filter("SVG", &["svg"])
                    .save_file()
                {
                    match self.engine.export_svg(&path) {
                        Ok(_) => {},
                        Err(e) => eprintln!("Failed to export SVG: {}", e),
                    }
                }
            }

            ui.separator();

            // Save DNA
            if ui
                .button("💾 Save DNA")
                .on_hover_text("Save the current triangle DNA to a JSON file")
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .set_file_name("dna.json")
                    .add_filter("JSON", &["json"])
                    .save_file()
                {
                    match self.engine.save_dna_to_file(&path) {
                        Ok(_) => {},
                        Err(e) => eprintln!("Failed to save DNA: {}", e),
                    }
                }
            }

            // Load DNA
            if ui
                .button("📂 Load DNA")
                .on_hover_text("Load triangle DNA from a JSON file (replaces current DNA)")
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("JSON", &["json"])
                    .pick_file()
                {
                    match self.engine.load_dna_from_file(&path) {
                        Ok(_) => {
                            // Force re-upload of canvas on next frame
                            self.last_uploaded_generation = 0;
                        },
                        Err(e) => eprintln!("Failed to load DNA: {}", e),
                    }
                }
            }

            ui.separator();

            // Run / Pause toggle
            let is_running = self.engine.is_running();
            let btn_label = if is_running { "⏸ Pause" } else { "▶ Run" };
            if ui.button(btn_label).clicked() {
                self.engine.toggle_running_state();
            }

            // Canvas-only view toggle
            ui.checkbox(&mut self.show_canvas_only, "Canvas only");

            ui.separator();

            // ------------- Canvas color picker (clears and re-baselines) -------------
            // Use egui's sRGBA color edit button and push the change into the engine.
            ui.label("Canvas color");
            {
                // Create the color-edit button (egui 0.27 requires Alpha parameter)
                let mut resp = egui::color_picker::color_edit_button_srgba(
                    ui,
                    &mut self.canvas_clear_color,
                    egui::color_picker::Alpha::BlendOrAdditive,
                );
            
                // on_hover_text(self) consumes the Response and returns a new one.
                resp = resp.on_hover_text(
                    "Background color for a...\nChanging this clears the canvas and restarts the baseline."
                );
            
                if resp.changed() {
                    let [r, g, b, a] = self.canvas_clear_color.to_array();
                    self.engine.set_background_color_and_clear([r, g, b, a]);
                    self.last_uploaded_generation = 0;
                }
            }

            ui.separator();

            // ------------- Max triangles (live engine setting, logarithmic) -------------
            let current_cap = self.engine.max_triangles();
            let mut cap_ui_value = current_cap as u32;
            let slider = egui::Slider::new(&mut cap_ui_value, 16..=100_000)
                .logarithmic(true)
                .text("Max triangles");
            let resp = ui.add(slider);
            resp.on_hover_text(
                "Upper bound on triangle DNA length.\n\
                 Lower = faster; higher = more detail.",
            );
            if cap_ui_value as usize != current_cap {
                self.engine.set_max_triangles(cap_ui_value as usize);
            }
            // Fixed-width readout prevents jitter.
            ui.monospace(format!("cap: {:>5}", cap_ui_value));

            ui.separator();

            // ------------- Stats window toggle -------------
            ui.checkbox(&mut self.show_stats_window, "📊 Show Stats");

            ui.separator();

            // ------------- Compute Backend Selector -------------
            ui.label("Compute:");
            let current_backend = self.engine.get_compute_backend();

            // CPU option (always available)
            let cpu_selected = ui.selectable_label(
                current_backend == ComputeBackendType::Cpu,
                "CPU"
            ).clicked();

            // WGPU GPU option (only if available)
            let wgpu_available = ComputeBackendType::Wgpu.is_available();
            let wgpu_label = if wgpu_available { "WGPU GPU" } else { "GPU (N/A)" };

            let mut wgpu_selected = false;
            ui.add_enabled_ui(wgpu_available, |ui| {
                wgpu_selected = ui.selectable_label(
                    current_backend == ComputeBackendType::Wgpu,
                    wgpu_label
                ).clicked();
            });

            // Handle selection changes
            if cpu_selected {
                self.engine.set_compute_backend(ComputeBackendType::Cpu);
            } else if wgpu_selected && wgpu_available {
                self.engine.set_compute_backend(ComputeBackendType::Wgpu);
            }
        });
    }

}

impl eframe::App for MiraiTrace {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        // Upload the latest canvas (if generation advanced) before painting UI.
        self.maybe_upload_canvas(ctx);

        // ---------- Top bar ----------
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            self.top_bar(ctx, ui);
        });

        // ---------- Central panel (strict left/right split) ----------
        egui::CentralPanel::default().show(ctx, |ui| {
            // Determine aspect-fit size to use.
            let (canvas_w, canvas_h) = {
                if let Some(tex) = &self.canvas_texture {
                    let [w, h] = tex.size();
                    (w as usize, h as usize)
                } else if self.original_width > 0 {
                    (self.original_width, self.original_height)
                } else {
                    let avail = ui.available_size();
                    (avail.x as usize, avail.y as usize)
                }
            };

            if self.show_canvas_only {
                // Canvas fills the central panel while maintaining aspect ratio.
                if let Some(tex) = &self.canvas_texture {
                    Self::draw_image_aspect_fit(ui, tex, canvas_w, canvas_h);
                } else {
                    ui.centered_and_justified(|ui| ui.label("Load an image to begin"));
                }
            } else {
                // Strict 2-column split: left = Canvas, right = Original.
                ui.columns(2, |columns| {
                    // Left: Canvas
                    egui::Frame::none()
                        .inner_margin(Margin::symmetric(4.0, 4.0))
                        .show(&mut columns[0], |ui| {
                            ui.vertical_centered(|ui| {
                                if let Some(tex) = &self.canvas_texture {
                                    Self::draw_image_aspect_fit(ui, tex, canvas_w, canvas_h);
                                } else {
                                    ui.centered_and_justified(|ui| ui.label("Canvas (no content yet)"));
                                }
                            });
                        });

                    // Right: Original
                    egui::Frame::none()
                        .inner_margin(Margin::symmetric(4.0, 4.0))
                        .show(&mut columns[1], |ui| {
                            ui.vertical_centered(|ui| {
                                if let Some(tex) = &self.original_texture {
                                    Self::draw_image_aspect_fit(ui, tex, self.original_width, self.original_height);
                                } else {
                                    ui.centered_and_justified(|ui| ui.label("Original (load an image)"));
                                }
                            });
                        });
                });
            }
        });

        // ---------- Floating Stats Window ----------
        if self.show_stats_window {
            egui::Window::new("📊 Statistics")
                .default_pos([10.0, 60.0])
                .default_size([450.0, 350.0])
                .resizable(true)
                .collapsible(true)
                .show(ctx, |ui| {
                    self.stats_window_content(ui);
                });
        }

        ctx.request_repaint_after(std::time::Duration::from_millis(33));
    }
}

impl MiraiTrace {
    /// Content for the floating stats window
    fn stats_window_content(&mut self, ui: &mut egui::Ui) {
        let stats: EvolutionStats = self.engine.capture_stats_snapshot();

        ui.style_mut().override_text_style = Some(egui::TextStyle::Monospace);

        // --- Backend Indicator (Prominent) ---
        ui.horizontal(|ui| {
            ui.label("Backend:");

            // Color-code based on backend type
            let (text, color) = if stats.active_backend.starts_with("WGPU") {
                (stats.active_backend.clone(), egui::Color32::from_rgb(50, 200, 50)) // Green for GPU
            } else if stats.active_backend.contains("fallback") || stats.active_backend.contains("failed") {
                (stats.active_backend.clone(), egui::Color32::from_rgb(255, 165, 0)) // Orange for fallback
            } else {
                (stats.active_backend.clone(), egui::Color32::GRAY) // Gray for CPU
            };

            ui.colored_label(color, text);
        });

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);

        // --- Core Evolution Stats ---
        ui.heading("Evolution Progress");
        ui.separator();

        let tri = stats.triangle_count;
        let gen = stats.generation_counter;
        let curr_err = stats.current_error.unwrap_or(0);
        let best_err = stats.best_error.unwrap_or(0);
        let delta = stats.last_accept_delta.unwrap_or(0);

        ui.label(format!("Triangles:     {:>6}", tri));
        ui.label(format!("Generation:    {:>8}", gen));
        ui.add_space(4.0);
        ui.label(format!("Current error: {:>12}", curr_err));
        ui.label(format!("Best error:    {:>12}", best_err));
        ui.label(format!("Last delta:    {:>8}", delta));

        ui.add_space(12.0);

        // --- Performance Metrics ---
        ui.heading("Performance");
        ui.separator();

        let pps = stats.proposals_per_second;
        let aps = stats.accepts_per_second;
        let accept_pct = stats.recent_acceptance_percent;

        ui.label(format!("Proposals/sec: {:>6.1}", pps));
        ui.label(format!("Accepts/sec:   {:>6.1}", aps));
        ui.label(format!("Acceptance:    {:>5.1}%", accept_pct));

        ui.add_space(12.0);

        // --- Error History Plot ---
        if !stats.error_history.is_empty() {
            ui.heading("Error History");
            ui.separator();

            let plot_height = 140.0;
            let plot_response = ui.allocate_response(
                egui::vec2(ui.available_width(), plot_height),
                egui::Sense::hover(),
            );
            let rect = plot_response.rect;
            let painter = ui.painter_at(rect);

            let series = &stats.error_history;
            let min_v = *series.iter().min().unwrap();
            let max_v = *series.iter().max().unwrap();
            let span = (max_v.saturating_sub(min_v)).max(1);

            let n = series.len();
            for i in 1..n {
                let x0 = rect.left() + (i as f32 - 1.0) / (n - 1) as f32 * rect.width();
                let x1 = rect.left() + (i as f32) / (n - 1) as f32 * rect.width();

                let y0 = rect.bottom()
                    - ((series[i - 1] - min_v) as f32 / span as f32) * rect.height();
                let y1 = rect.bottom()
                    - ((series[i] - min_v) as f32 / span as f32) * rect.height();

                painter.line_segment(
                    [egui::pos2(x0, y0), egui::pos2(x1, y1)],
                    egui::Stroke::new(2.5, ui.visuals().text_color()),
                );
            }
        }
    }
}
