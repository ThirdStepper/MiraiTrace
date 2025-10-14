// -----------------------------------------------------------------------------
// Triangle DNA
// -----------------------------------------------------------------------------

use std::cmp::{max,min};
use serde::{Deserialize, Serialize};

use super::IntRect;

// repr(C) ensures memory layout matches GPU expectations
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub(crate) struct Triangle {
    pub x0: i32, pub y0: i32,
    pub x1: i32, pub y1: i32,
    pub x2: i32, pub y2: i32,
    pub r: u8,  pub g: u8,  pub b: u8,  pub a: u8,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct TriangleDna {
    pub triangles: Vec<Triangle>,
}

pub(crate) fn triangle_bbox_px(tri: &Triangle, canvas_w: usize, canvas_h: usize) -> IntRect {
    // Clamp all to canvas, then compute AABB
    let min_x = min(tri.x0.min(tri.x1).min(tri.x2).max(0), canvas_w as i32 - 1) as usize;
    let min_y = min(tri.y0.min(tri.y1).min(tri.y2).max(0), canvas_h as i32 - 1) as usize;
    let max_x = max(tri.x0.max(tri.x1).max(tri.x2).min(canvas_w as i32 - 1), 0) as usize;
    let max_y = max(tri.y0.max(tri.y1).max(tri.y2).min(canvas_h as i32 - 1), 0) as usize;
    if max_x < min_x || max_y < min_y {
        return IntRect::empty();
    }
    IntRect {
        x: min_x,
        y: min_y,
        w: max_x - min_x + 1,
        h: max_y - min_y + 1,
    }
}


