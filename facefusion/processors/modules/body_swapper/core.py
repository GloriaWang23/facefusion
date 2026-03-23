from argparse import ArgumentParser
from typing import List, Tuple

import cv2
import numpy

import facefusion.jobs.job_store
from facefusion import config, content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, logger, state_manager, translator, video_manager
from facefusion.face_analyser import get_many_faces, get_one_face
from facefusion.face_selector import select_faces
from facefusion.filesystem import has_image, in_directory, is_image, is_video, same_file_extension
from facefusion.processors.modules.body_swapper.types import BodySwapperInputs
from facefusion.processors.types import ProcessorOutputs
from facefusion.program_helper import find_argument_group
from facefusion.types import ApplyStateItem, Args, BoundingBox, Face, InferencePool, ProcessMode, VisionFrame
from facefusion.vision import read_static_image, read_static_video_frame


def get_inference_pool() -> InferencePool:
	return {}


def clear_inference_pool() -> None:
	pass


def register_args(program : ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument('--body-swapper-body-height-ratio', help = translator.get('help.body_height_ratio', __package__), type = float, default = config.get_float_value('processors', 'body_swapper_body_height_ratio', '7.5'))
		group_processors.add_argument('--body-swapper-body-width-ratio', help = translator.get('help.body_width_ratio', __package__), type = float, default = config.get_float_value('processors', 'body_swapper_body_width_ratio', '3.5'))
		group_processors.add_argument('--body-swapper-blend', help = translator.get('help.blend', __package__), type = int, default = config.get_int_value('processors', 'body_swapper_blend', '80'))
		facefusion.jobs.job_store.register_step_keys([ 'body_swapper_body_height_ratio', 'body_swapper_body_width_ratio', 'body_swapper_blend' ])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('body_swapper_body_height_ratio', args.get('body_swapper_body_height_ratio'))
	apply_state_item('body_swapper_body_width_ratio', args.get('body_swapper_body_width_ratio'))
	apply_state_item('body_swapper_blend', args.get('body_swapper_blend'))


def pre_check() -> bool:
	return True


def pre_process(mode : ProcessMode) -> bool:
	if not has_image(state_manager.get_item('source_paths')):
		logger.error(translator.get('choose_image_source') + translator.get('exclamation_mark'), __name__)
		return False
	if mode in [ 'output', 'preview' ] and not is_image(state_manager.get_item('target_path')) and not is_video(state_manager.get_item('target_path')):
		logger.error(translator.get('choose_image_or_video_target') + translator.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not in_directory(state_manager.get_item('output_path')):
		logger.error(translator.get('specify_image_or_video_output') + translator.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not same_file_extension(state_manager.get_item('target_path'), state_manager.get_item('output_path')):
		logger.error(translator.get('match_target_and_output_extension') + translator.get('exclamation_mark'), __name__)
		return False
	return True


def post_process() -> None:
	read_static_image.cache_clear()
	read_static_video_frame.cache_clear()
	video_manager.clear_video_pool()
	if state_manager.get_item('video_memory_strategy') == 'strict':
		content_analyser.clear_inference_pool()
		face_classifier.clear_inference_pool()
		face_detector.clear_inference_pool()
		face_landmarker.clear_inference_pool()
		face_masker.clear_inference_pool()
		face_recognizer.clear_inference_pool()


def estimate_body_bounding_box(face_bounding_box : BoundingBox, frame_shape : Tuple[int, ...]) -> BoundingBox:
	x1, y1, x2, y2 = face_bounding_box
	face_width = x2 - x1
	face_height = y2 - y1
	face_center_x = (x1 + x2) / 2

	body_height_ratio = state_manager.get_item('body_swapper_body_height_ratio') or 7.5
	body_width_ratio = state_manager.get_item('body_swapper_body_width_ratio') or 3.5

	body_height = face_height * body_height_ratio
	body_width = face_width * body_width_ratio

	# Include some forehead above the face bounding box
	body_top = y1 - face_height * 0.4
	body_bottom = body_top + body_height
	body_left = face_center_x - body_width / 2
	body_right = face_center_x + body_width / 2

	frame_h, frame_w = frame_shape[:2]
	body_top = max(0.0, body_top)
	body_left = max(0.0, body_left)
	body_bottom = min(float(frame_h), body_bottom)
	body_right = min(float(frame_w), body_right)

	return numpy.array([ body_left, body_top, body_right, body_bottom ])


def create_soft_body_mask(width : int, height : int, edge_ratio : float = 0.15) -> numpy.ndarray:
	mask = numpy.ones((height, width), dtype = numpy.float32)
	edge_x = max(1, int(width * edge_ratio))
	edge_y = max(1, int(height * edge_ratio))

	# Horizontal gradient at left and right edges
	for i in range(edge_x):
		alpha = i / edge_x
		mask[:, i] = numpy.minimum(mask[:, i], alpha)
		mask[:, width - 1 - i] = numpy.minimum(mask[:, width - 1 - i], alpha)

	# Vertical gradient at top and bottom edges
	for i in range(edge_y):
		alpha = i / edge_y
		mask[i, :] = numpy.minimum(mask[i, :], alpha)
		mask[height - 1 - i, :] = numpy.minimum(mask[height - 1 - i, :], alpha)

	return mask


def swap_body(source_vision_frame : VisionFrame, source_face : Face, target_face : Face, temp_vision_frame : VisionFrame) -> VisionFrame:
	source_body_box = estimate_body_bounding_box(source_face.bounding_box, source_vision_frame.shape)
	target_body_box = estimate_body_bounding_box(target_face.bounding_box, temp_vision_frame.shape)

	sx1, sy1, sx2, sy2 = int(source_body_box[0]), int(source_body_box[1]), int(source_body_box[2]), int(source_body_box[3])
	tx1, ty1, tx2, ty2 = int(target_body_box[0]), int(target_body_box[1]), int(target_body_box[2]), int(target_body_box[3])

	source_crop_w = sx2 - sx1
	source_crop_h = sy2 - sy1
	target_w = tx2 - tx1
	target_h = ty2 - ty1

	if source_crop_w <= 0 or source_crop_h <= 0 or target_w <= 0 or target_h <= 0:
		return temp_vision_frame

	source_body_crop = source_vision_frame[sy1:sy2, sx1:sx2]
	source_body_resized = cv2.resize(source_body_crop, (target_w, target_h), interpolation = cv2.INTER_LINEAR)

	blend = (state_manager.get_item('body_swapper_blend') or 80) / 100.0
	body_mask = create_soft_body_mask(target_w, target_h) * blend
	body_mask_3ch = numpy.stack([ body_mask ] * 3, axis = -1)

	temp_vision_frame = temp_vision_frame.copy()
	target_region = temp_vision_frame[ty1:ty2, tx1:tx2].astype(numpy.float32)
	source_body_resized = source_body_resized.astype(numpy.float32)
	blended = source_body_resized * body_mask_3ch + target_region * (1 - body_mask_3ch)
	temp_vision_frame[ty1:ty2, tx1:tx2] = numpy.clip(blended, 0, 255).astype(numpy.uint8)

	return temp_vision_frame


def process_frame(inputs : BodySwapperInputs) -> ProcessorOutputs:
	reference_vision_frame = inputs.get('reference_vision_frame')
	source_vision_frames = inputs.get('source_vision_frames')
	target_vision_frame = inputs.get('target_vision_frame')
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')

	if not source_vision_frames:
		return temp_vision_frame, temp_vision_mask

	# Find source face (use the largest face in the source images)
	source_faces = []
	for source_frame in source_vision_frames:
		faces = get_many_faces([ source_frame ])
		if faces:
			source_faces.extend(faces)

	source_face = get_one_face(source_faces)

	if not source_face:
		return temp_vision_frame, temp_vision_mask

	# Use the first source frame that contains the source face
	source_vision_frame = None
	for frame in source_vision_frames:
		if get_one_face(get_many_faces([ frame ])):
			source_vision_frame = frame
			break

	if source_vision_frame is None:
		return temp_vision_frame, temp_vision_mask

	target_faces = select_faces(reference_vision_frame, target_vision_frame)

	if target_faces:
		for target_face in target_faces:
			temp_vision_frame = swap_body(source_vision_frame, source_face, target_face, temp_vision_frame)

	return temp_vision_frame, temp_vision_mask
