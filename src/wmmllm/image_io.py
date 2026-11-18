"""Decode color once and encode only reconstructed pixels, without metadata."""
from __future__ import annotations

import hashlib
import io
import struct
from pathlib import Path
from typing import Any

from PIL import Image, ImageCms


def _color_tags(profile: bytes) -> tuple[bytes, tuple[tuple[bytes, bytes], ...]]:
    """Compare transform data, excluding profile identity and descriptive tags."""
    descriptive_tags = {b"desc", b"cprt", b"dmnd", b"dmdd", b"vued", b"meta"}
    if len(profile) < 132:
        raise ValueError("Invalid ICC profile")
    count = struct.unpack_from(">I", profile, 128)[0]
    if count > (len(profile) - 132) // 12:
        raise ValueError("Invalid ICC tag table")
    tags = []
    for index in range(count):
        start = 132 + 12 * index
        name, offset, size = struct.unpack_from(">4sII", profile, start)
        if name not in descriptive_tags:
            if offset + size > len(profile):
                raise ValueError("Invalid ICC color tag")
            tags.append((name, profile[offset:offset + size]))
    return profile[12:24] + profile[68:80], tuple(sorted(tags))


def prepare_rgb_image(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    """Return detached RGB pixels and a record containing no source metadata.

    Embedded ICC profiles are applied once, before metadata is discarded. An
    invalid or incompatible profile fails ingestion instead of silently changing
    its color interpretation. EXIF orientation is not applied: decoded pixel
    coordinates retain their existing meaning throughout the geometry pipeline.
    """
    image.load()
    original = image.convert("RGB")
    pixels = original
    profile = image.info.get("icc_profile")
    conversion = "none"
    if profile:
        try:
            source_profile = ImageCms.ImageCmsProfile(io.BytesIO(profile))
            target_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
            if _color_tags(source_profile.tobytes()) == _color_tags(target_profile.tobytes()):
                conversion = "srgb_identity"
            else:
                # Conversion happens in the profile's own input color space.
                source = image if image.mode in ("RGB", "CMYK", "LAB", "L") else original
                pixels = ImageCms.profileToProfile(
                    source, source_profile, target_profile,
                    outputMode="RGB", renderingIntent=ImageCms.Intent.PERCEPTUAL)
                conversion = "embedded_icc_to_srgb"
        except (TypeError, ValueError, OSError, ImageCms.PyCMSError):
            raise ValueError("Image color profile could not be converted to sRGB") from None
    raw = pixels.tobytes()
    clean = Image.frombytes("RGB", pixels.size, raw)
    record = {"schema": "wmmllm.image_conversion.v1",
              "color_conversion": conversion,
              "decoded_rgb_pixels_changed": raw != original.tobytes(),
              "pixel_sha256": hashlib.sha256(raw).hexdigest()}
    return clean, record


def load_rgb_image(path: str | Path) -> tuple[Image.Image, dict[str, Any]]:
    with Image.open(path) as image:
        return prepare_rgb_image(image)


def encode_rgb_png(image: Image.Image) -> tuple[bytes, dict[str, Any]]:
    """Losslessly encode reconstructed RGB pixels and verify decoded equality."""
    clean, record = prepare_rgb_image(image)
    stream = io.BytesIO()
    clean.save(stream, format="PNG")
    encoded = stream.getvalue()
    with Image.open(io.BytesIO(encoded)) as decoded:
        decoded.load()
        if (decoded.mode != "RGB" or decoded.size != clean.size
                or decoded.tobytes() != clean.tobytes() or decoded.info
                or len(decoded.getexif())):
            raise ValueError("Image PNG round trip changed pixels or retained metadata")
    return encoded, record

