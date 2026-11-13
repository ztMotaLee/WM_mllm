from pathlib import Path

from wmmllm.providers import (NoneVideoBackend, PresentationResult, VideoBackend,
                             VideoTransition)


def test_none_backend_never_claims_rendered_or_generated_video(tmp_path):
    first, last = tmp_path / "first.png", tmp_path / "last.png"
    transition = VideoTransition("transition-1", "local_edit", first, last,
                                 duration_seconds=5, output_path=tmp_path / "video.mp4")
    backend = NoneVideoBackend()
    result = backend.present(transition)
    assert isinstance(backend, VideoBackend)
    assert result.status == "none" and result.video_path is None and result.backend == "none"
    assert not list(tmp_path.iterdir())


def test_result_distinguishes_cpu_rendering_from_generation():
    rendered = PresentationResult("rendered", Path("rotation.mp4"), "cpu_panorama")
    generated = PresentationResult("generated", Path("transition.mp4"), "external")
    assert rendered.status != generated.status
