#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry for the packaged static-world exporter."""
from wmmllm.export_viewer import ExportError, export_world, main

if __name__ == "__main__":
    raise SystemExit(main())
