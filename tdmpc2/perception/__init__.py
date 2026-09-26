"""Frozen perception adapters used by standalone diagnostics.

Nothing in this package is imported by the TD-MPC2 training path.  Perception
backends must be selected explicitly so a missing dependency can never change
an experiment into a different, silent fallback.
"""

from .cutie_oc_adapter import (
	CutieDependencyError,
	CutieFrameResult,
	CutieOCAdapter,
	CutieOCConfig,
	CutiePreflightError,
	CutieSupportPrompts,
	inspect_cutie_installation,
	load_point_support_prompts,
)
from .visual_small_cutie_decoder import (
	VisualSmallCutieCalibration,
	VisualSmallCutieDecodeResult,
	VisualSmallCutieDecoderContractError,
	VisualSmallCutiePointDecoder,
	load_visual_small_cutie_calibration,
)
from .visual_small_whole_arm_decoder import (
	VisualSmallWholeArmCalibration,
	VisualSmallWholeArmDecodeResult,
	VisualSmallWholeArmDecoderContractError,
	VisualSmallWholeArmPointDecoder,
	load_visual_small_whole_arm_calibration,
)

__all__ = (
	'CutieDependencyError',
	'CutieFrameResult',
	'CutieOCAdapter',
	'CutieOCConfig',
	'CutiePreflightError',
	'CutieSupportPrompts',
	'inspect_cutie_installation',
	'load_point_support_prompts',
	'VisualSmallCutieCalibration',
	'VisualSmallCutieDecodeResult',
	'VisualSmallCutieDecoderContractError',
	'VisualSmallCutiePointDecoder',
	'load_visual_small_cutie_calibration',
	'VisualSmallWholeArmCalibration',
	'VisualSmallWholeArmDecodeResult',
	'VisualSmallWholeArmDecoderContractError',
	'VisualSmallWholeArmPointDecoder',
	'load_visual_small_whole_arm_calibration',
)
