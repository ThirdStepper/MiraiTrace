//! Program entry point: constructs the engine, configures the window,
//! and launches the egui/eframe UI.

mod engine;
mod ui;

use eframe::{egui, NativeOptions};
use engine::{EvolutionEngine, FrameDimensions};
use ui::MiraiTrace;

fn main() -> eframe::Result<()> {
    // Initial canvas size (will resize to target image on open).
    let initial_dimensions = FrameDimensions { width: 1024, height: 768 };
    let evolution_engine = EvolutionEngine::new(initial_dimensions);

    // eframe 0.27: configure via ViewportBuilder
    let native_options = NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(egui::vec2(
                initial_dimensions.width as f32,
                initial_dimensions.height as f32,
            ))
            .with_title("MiraiTrace"),
        ..Default::default()
    };

    eframe::run_native(
        "MiraiTrace",
        native_options,
        Box::new(move |_cc| Box::new(MiraiTrace::new(evolution_engine))),
    )
}
