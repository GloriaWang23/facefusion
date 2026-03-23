from facefusion.types import Locales

LOCALES : Locales =\
{
	'en':
	{
		'help':
		{
			'model': 'choose the model responsible for swapping the face',
			'pixel_boost': 'choose the pixel boost resolution for the face swapper',
			'weight': 'specify the degree of weight applied to the face',
			'warp_template': 'override the warp template used to crop the face region (use head_256 for full head replacement)'
		},
		'uis':
		{
			'model_dropdown': 'FACE SWAPPER MODEL',
			'pixel_boost_dropdown': 'FACE SWAPPER PIXEL BOOST',
			'weight_slider': 'FACE SWAPPER WEIGHT',
			'warp_template_dropdown': 'WARP TEMPLATE'
		}
	}
}
