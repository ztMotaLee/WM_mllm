"""Presentation interface. CPU panorama rendering belongs to the core engine."""
from __future__ import annotations

from .types import PresentationResult, VideoTransition


class NoneVideoBackend:
    def present(self, transition: VideoTransition) -> PresentationResult:
        return PresentationResult(status="none", video_path=None, backend="none",
                                  reason="Generated transition video is unavailable.")
