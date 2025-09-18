// src/engine/geom.rs
use std::cmp::{max,min};
use super::IntRect;

#[inline]
pub(crate) fn clamp_i32(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

pub(crate) fn union_rect(a: &IntRect, b: &IntRect) -> IntRect {
    if a.w == 0 || a.h == 0 {
        return *b;
    }
    if b.w == 0 || b.h == 0 {
        return *a;
    }
    let x0 = min(a.x, b.x);
    let y0 = min(a.y, b.y);
    let x1 = max(a.x + a.w, b.x + b.w);
    let y1 = max(a.y + a.h, b.y + b.h);
    IntRect {
        x: x0,
        y: y0,
        w: x1 - x0,
        h: y1 - y0,
    }
}

#[inline]
pub(crate) fn pad_and_clamp_rect(mut r: IntRect, pad: usize, w: usize, h: usize) -> IntRect {
    // Left/top after padding (saturate at 0)
    let x0 = r.x.saturating_sub(pad);
    let y0 = r.y.saturating_sub(pad);

    // Right/bottom after padding, clamped to image bounds
    let x1 = (r.x + r.w).saturating_add(pad).min(w);
    let y1 = (r.y + r.h).saturating_add(pad).min(h);

    // Write back clamped rect
    r.x = x0;
    r.y = y0;
    r.w = x1.saturating_sub(x0);
    r.h = y1.saturating_sub(y0);
    r
}