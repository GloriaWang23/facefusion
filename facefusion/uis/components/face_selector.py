from typing import Any, List, Optional, Tuple

import cv2
import gradio
from gradio_rangeslider import RangeSlider

import facefusion.choices
from facefusion import state_manager, translator
from facefusion.common_helper import calculate_float_step, calculate_int_step
from facefusion.face_analyser import get_many_faces
from facefusion.face_selector import sort_and_filter_faces
from facefusion.face_store import clear_static_faces
from facefusion.filesystem import filter_image_paths, is_image, is_video
from facefusion.types import FaceSelectorMode, FaceSelectorOrder, Gender, Race, VisionFrame
from facefusion.uis.core import get_ui_component, get_ui_components, register_ui_component
from facefusion.uis.types import ComponentOptions
from facefusion.uis.ui_helper import convert_str_none
from facefusion.vision import fit_cover_frame, read_static_image, read_video_frame

FACE_SWAP_PAIR_COUNT = 4

FACE_SELECTOR_MODE_DROPDOWN : Optional[gradio.Dropdown] = None
FACE_SELECTOR_ORDER_DROPDOWN : Optional[gradio.Dropdown] = None
FACE_SELECTOR_GENDER_DROPDOWN : Optional[gradio.Dropdown] = None
FACE_SELECTOR_RACE_DROPDOWN : Optional[gradio.Dropdown] = None
FACE_SELECTOR_AGE_RANGE_SLIDER : Optional[RangeSlider] = None
REFERENCE_FACE_POSITION_GALLERY : Optional[gradio.Gallery] = None
REFERENCE_FACE_DISTANCE_SLIDER : Optional[gradio.Slider] = None
FACE_MAP_PAIRS_GROUP : Optional[gradio.Group] = None
FACE_MAP_PAIR_POSITION_NUMBERS : List[gradio.Number] = []
FACE_MAP_PAIR_SOURCE_FILES : List[gradio.File] = []


def render() -> None:
	global FACE_SELECTOR_MODE_DROPDOWN
	global FACE_SELECTOR_ORDER_DROPDOWN
	global FACE_SELECTOR_GENDER_DROPDOWN
	global FACE_SELECTOR_RACE_DROPDOWN
	global FACE_SELECTOR_AGE_RANGE_SLIDER
	global REFERENCE_FACE_POSITION_GALLERY
	global REFERENCE_FACE_DISTANCE_SLIDER
	global FACE_MAP_PAIRS_GROUP
	global FACE_MAP_PAIR_POSITION_NUMBERS
	global FACE_MAP_PAIR_SOURCE_FILES

	FACE_MAP_PAIR_POSITION_NUMBERS = []
	FACE_MAP_PAIR_SOURCE_FILES = []

	is_map_mode = state_manager.get_item('face_selector_mode') == 'map'
	is_reference_or_map_mode = state_manager.get_item('face_selector_mode') in [ 'reference', 'map' ]

	reference_face_gallery_options : ComponentOptions =\
	{
		'label': translator.get('uis.reference_face_gallery'),
		'object_fit': 'cover',
		'columns': 7,
		'allow_preview': False,
		'elem_classes': 'box-face-selector',
		'visible': is_reference_or_map_mode
	}
	if is_image(state_manager.get_item('target_path')):
		target_vision_frame = read_static_image(state_manager.get_item('target_path'))
		reference_face_gallery_options['value'] = extract_gallery_frames(target_vision_frame)
	if is_video(state_manager.get_item('target_path')):
		target_vision_frame = read_video_frame(state_manager.get_item('target_path'), state_manager.get_item('reference_frame_number'))
		reference_face_gallery_options['value'] = extract_gallery_frames(target_vision_frame)

	FACE_SELECTOR_MODE_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.face_selector_mode_dropdown'),
		choices = facefusion.choices.face_selector_modes,
		value = state_manager.get_item('face_selector_mode')
	)
	REFERENCE_FACE_POSITION_GALLERY = gradio.Gallery(**reference_face_gallery_options)

	existing_pairs = state_manager.get_item('face_swap_pairs') or []
	with gradio.Group(visible = is_map_mode) as FACE_MAP_PAIRS_GROUP:
		for pair_index in range(FACE_SWAP_PAIR_COUNT):
			existing_pair = existing_pairs[pair_index] if pair_index < len(existing_pairs) else None
			with gradio.Row():
				position_number = gradio.Number(
					label = translator.get('uis.face_map_pair_position_number') + ' ' + str(pair_index + 1),
					value = existing_pair.get('reference_face_position', pair_index) if existing_pair else pair_index,
					minimum = 0,
					maximum = 99,
					precision = 0
				)
				source_file = gradio.File(
					label = translator.get('uis.face_map_pair_source_file') + ' ' + str(pair_index + 1),
					file_count = 'multiple',
					file_types = [ 'image' ],
					value = existing_pair.get('source_paths') if existing_pair else None
				)
			FACE_MAP_PAIR_POSITION_NUMBERS.append(position_number)
			FACE_MAP_PAIR_SOURCE_FILES.append(source_file)

	with gradio.Group():
		with gradio.Row():
			FACE_SELECTOR_ORDER_DROPDOWN = gradio.Dropdown(
				label = translator.get('uis.face_selector_order_dropdown'),
				choices = facefusion.choices.face_selector_orders,
				value = state_manager.get_item('face_selector_order')
			)
			FACE_SELECTOR_GENDER_DROPDOWN = gradio.Dropdown(
				label = translator.get('uis.face_selector_gender_dropdown'),
				choices = [ 'none' ] + facefusion.choices.face_selector_genders,
				value = state_manager.get_item('face_selector_gender') or 'none'
			)
			FACE_SELECTOR_RACE_DROPDOWN = gradio.Dropdown(
				label = translator.get('uis.face_selector_race_dropdown'),
				choices = [ 'none' ] + facefusion.choices.face_selector_races,
				value = state_manager.get_item('face_selector_race') or 'none'
			)
		with gradio.Row():
			face_selector_age_start = state_manager.get_item('face_selector_age_start') or facefusion.choices.face_selector_age_range[0]
			face_selector_age_end = state_manager.get_item('face_selector_age_end') or facefusion.choices.face_selector_age_range[-1]
			FACE_SELECTOR_AGE_RANGE_SLIDER = RangeSlider(
				label = translator.get('uis.face_selector_age_range_slider'),
				minimum = facefusion.choices.face_selector_age_range[0],
				maximum = facefusion.choices.face_selector_age_range[-1],
				value = (face_selector_age_start, face_selector_age_end),
				step = calculate_int_step(facefusion.choices.face_selector_age_range)
			)
	REFERENCE_FACE_DISTANCE_SLIDER = gradio.Slider(
		label = translator.get('uis.reference_face_distance_slider'),
		value = state_manager.get_item('reference_face_distance'),
		step = calculate_float_step(facefusion.choices.reference_face_distance_range),
		minimum = facefusion.choices.reference_face_distance_range[0],
		maximum = facefusion.choices.reference_face_distance_range[-1],
		visible = is_reference_or_map_mode
	)
	register_ui_component('face_selector_mode_dropdown', FACE_SELECTOR_MODE_DROPDOWN)
	register_ui_component('face_selector_order_dropdown', FACE_SELECTOR_ORDER_DROPDOWN)
	register_ui_component('face_selector_gender_dropdown', FACE_SELECTOR_GENDER_DROPDOWN)
	register_ui_component('face_selector_race_dropdown', FACE_SELECTOR_RACE_DROPDOWN)
	register_ui_component('face_selector_age_range_slider', FACE_SELECTOR_AGE_RANGE_SLIDER)
	register_ui_component('reference_face_position_gallery', REFERENCE_FACE_POSITION_GALLERY)
	register_ui_component('reference_face_distance_slider', REFERENCE_FACE_DISTANCE_SLIDER)


def listen() -> None:
	pair_inputs = []
	for pair_index in range(FACE_SWAP_PAIR_COUNT):
		pair_inputs.append(FACE_MAP_PAIR_POSITION_NUMBERS[pair_index])
		pair_inputs.append(FACE_MAP_PAIR_SOURCE_FILES[pair_index])

	FACE_SELECTOR_MODE_DROPDOWN.change(update_face_selector_mode, inputs = FACE_SELECTOR_MODE_DROPDOWN, outputs = [ REFERENCE_FACE_POSITION_GALLERY, REFERENCE_FACE_DISTANCE_SLIDER, FACE_MAP_PAIRS_GROUP ])
	FACE_SELECTOR_ORDER_DROPDOWN.change(update_face_selector_order, inputs = FACE_SELECTOR_ORDER_DROPDOWN, outputs = REFERENCE_FACE_POSITION_GALLERY)
	FACE_SELECTOR_GENDER_DROPDOWN.change(update_face_selector_gender, inputs = FACE_SELECTOR_GENDER_DROPDOWN, outputs = REFERENCE_FACE_POSITION_GALLERY)
	FACE_SELECTOR_RACE_DROPDOWN.change(update_face_selector_race, inputs = FACE_SELECTOR_RACE_DROPDOWN, outputs = REFERENCE_FACE_POSITION_GALLERY)
	FACE_SELECTOR_AGE_RANGE_SLIDER.release(update_face_selector_age_range, inputs = FACE_SELECTOR_AGE_RANGE_SLIDER, outputs = REFERENCE_FACE_POSITION_GALLERY)
	REFERENCE_FACE_DISTANCE_SLIDER.release(update_reference_face_distance, inputs = REFERENCE_FACE_DISTANCE_SLIDER)

	for pair_index in range(FACE_SWAP_PAIR_COUNT):
		FACE_MAP_PAIR_POSITION_NUMBERS[pair_index].change(update_face_swap_pairs, inputs = pair_inputs)
		FACE_MAP_PAIR_SOURCE_FILES[pair_index].change(update_face_swap_pairs, inputs = pair_inputs)

	preview_frame_slider = get_ui_component('preview_frame_slider')
	if preview_frame_slider:
		REFERENCE_FACE_POSITION_GALLERY.select(update_reference_frame_number, inputs = preview_frame_slider)
		REFERENCE_FACE_POSITION_GALLERY.select(update_reference_face_position)

	for ui_component in get_ui_components(
	[
		'target_image',
		'target_video'
	]):
		for method in [ 'change', 'clear' ]:
			getattr(ui_component, method)(clear_reference_frame_number)
			getattr(ui_component, method)(clear_reference_face_position)
			getattr(ui_component, method)(update_reference_position_gallery, outputs = REFERENCE_FACE_POSITION_GALLERY)

	for ui_component in get_ui_components(
	[
		'face_detector_model_dropdown',
		'face_detector_size_dropdown',
		'face_detector_angles_checkbox_group'
	]):
		ui_component.change(clear_and_update_reference_position_gallery, outputs = REFERENCE_FACE_POSITION_GALLERY)

	face_detector_score_slider = get_ui_component('face_detector_score_slider')
	if face_detector_score_slider:
		face_detector_score_slider.release(update_reference_position_gallery, outputs = REFERENCE_FACE_POSITION_GALLERY)

	preview_frame_slider = get_ui_component('preview_frame_slider')
	if preview_frame_slider:
		for method in [ 'change', 'release' ]:
			getattr(preview_frame_slider, method)(update_reference_position_gallery, inputs = preview_frame_slider, outputs = REFERENCE_FACE_POSITION_GALLERY, show_progress = 'hidden')


def update_face_selector_mode(face_selector_mode : FaceSelectorMode) -> Tuple[gradio.Gallery, gradio.Slider, gradio.Group]:
	state_manager.set_item('face_selector_mode', face_selector_mode)
	if face_selector_mode == 'many':
		return gradio.Gallery(visible = False), gradio.Slider(visible = False), gradio.Group(visible = False)
	if face_selector_mode == 'one':
		return gradio.Gallery(visible = False), gradio.Slider(visible = False), gradio.Group(visible = False)
	if face_selector_mode == 'reference':
		return gradio.Gallery(visible = True), gradio.Slider(visible = True), gradio.Group(visible = False)
	if face_selector_mode == 'map':
		return gradio.Gallery(visible = True), gradio.Slider(visible = True), gradio.Group(visible = True)
	return gradio.Gallery(visible = False), gradio.Slider(visible = False), gradio.Group(visible = False)


def update_face_swap_pairs(*args : Any) -> None:
	face_swap_pairs = []
	for pair_index in range(FACE_SWAP_PAIR_COUNT):
		position_value = args[pair_index * 2]
		source_files = args[pair_index * 2 + 1]
		source_paths = [ file.name for file in source_files ] if source_files else []
		source_paths = filter_image_paths(source_paths)
		if source_paths:
			face_swap_pairs.append(
			{
				'source_paths': source_paths,
				'reference_face_position': int(position_value) if position_value is not None else pair_index
			})
	state_manager.set_item('face_swap_pairs', face_swap_pairs if face_swap_pairs else None)


def update_face_selector_order(face_analyser_order : FaceSelectorOrder) -> gradio.Gallery:
	state_manager.set_item('face_selector_order', convert_str_none(face_analyser_order))
	return update_reference_position_gallery()


def update_face_selector_gender(face_selector_gender : Gender) -> gradio.Gallery:
	state_manager.set_item('face_selector_gender', convert_str_none(face_selector_gender))
	return update_reference_position_gallery()


def update_face_selector_race(face_selector_race : Race) -> gradio.Gallery:
	state_manager.set_item('face_selector_race', convert_str_none(face_selector_race))
	return update_reference_position_gallery()


def update_face_selector_age_range(face_selector_age_range : Tuple[float, float]) -> gradio.Gallery:
	face_selector_age_start, face_selector_age_end = face_selector_age_range
	state_manager.set_item('face_selector_age_start', int(face_selector_age_start))
	state_manager.set_item('face_selector_age_end', int(face_selector_age_end))
	return update_reference_position_gallery()


def update_reference_face_position(event : gradio.SelectData) -> None:
	state_manager.set_item('reference_face_position', event.index)


def clear_reference_face_position() -> None:
	state_manager.set_item('reference_face_position', 0)


def update_reference_face_distance(reference_face_distance : float) -> None:
	state_manager.set_item('reference_face_distance', reference_face_distance)


def update_reference_frame_number(reference_frame_number : int = 0) -> None:
	state_manager.set_item('reference_frame_number', reference_frame_number)


def clear_reference_frame_number() -> None:
	state_manager.set_item('reference_frame_number', 0)


def clear_and_update_reference_position_gallery() -> gradio.Gallery:
	clear_static_faces()
	return update_reference_position_gallery()


def update_reference_position_gallery(frame_number : int = 0) -> gradio.Gallery:
	gallery_vision_frames = []
	if is_image(state_manager.get_item('target_path')):
		target_vision_frame = read_static_image(state_manager.get_item('target_path'))
		gallery_vision_frames = extract_gallery_frames(target_vision_frame)
	if is_video(state_manager.get_item('target_path')):
		target_vision_frame = read_video_frame(state_manager.get_item('target_path'), frame_number)
		gallery_vision_frames = extract_gallery_frames(target_vision_frame)
	if gallery_vision_frames:
		return gradio.Gallery(value = gallery_vision_frames)
	return gradio.Gallery(value = None)


def extract_gallery_frames(target_vision_frame : VisionFrame) -> List[VisionFrame]:
	gallery_vision_frames = []
	faces = get_many_faces([ target_vision_frame ])
	faces = sort_and_filter_faces(faces)

	for face in faces:
		start_x, start_y, end_x, end_y = map(int, face.bounding_box)
		padding_x = int((end_x - start_x) * 0.25)
		padding_y = int((end_y - start_y) * 0.25)
		start_x = max(0, start_x - padding_x)
		start_y = max(0, start_y - padding_y)
		end_x = max(0, end_x + padding_x)
		end_y = max(0, end_y + padding_y)
		crop_vision_frame = target_vision_frame[start_y:end_y, start_x:end_x]
		crop_vision_frame = fit_cover_frame(crop_vision_frame, (128, 128))
		crop_vision_frame = cv2.cvtColor(crop_vision_frame, cv2.COLOR_BGR2RGB)
		gallery_vision_frames.append(crop_vision_frame)
	return gallery_vision_frames
