/// Dimensions used when (re)allocating the canvas.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameDimensions {
    pub width: usize,
    pub height: usize,
}

/// Integer rectangle in pixel space.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct IntRect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

impl IntRect {
    #[inline]
    pub(crate) fn empty() -> Self {
        Self { x: 0, y: 0, w: 0, h: 0 }
    }
}