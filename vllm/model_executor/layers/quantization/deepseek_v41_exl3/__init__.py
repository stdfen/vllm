# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed FP8/EXL3 checkpoint compatibility, registered lazily by vLLM.

Ported from the archived deployment plugin. Optional CPU pinning, RDMA,
router and mHC plugin installation hooks are not installed by this module.
"""
