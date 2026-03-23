from typing import List, TypedDict

from facefusion.types import Mask, VisionFrame

BodySwapperInputs = TypedDict('BodySwapperInputs',
{
	'reference_vision_frame' : VisionFrame,
	'source_vision_frames' : List[VisionFrame],
	'target_vision_frame' : VisionFrame,
	'temp_vision_frame' : VisionFrame,
	'temp_vision_mask' : Mask
})
