from .double_inputs_functions.double_inputs_concatenated_hl import double_inputs_concatenated_hl
from .double_inputs_functions.double_inputs_concatenated_inputs import double_inputs_concatenated_inputs

def double_inputs(inputs_dimensions, output_dimension, first_layer, 
				  concatenate_axis, scale_factor, depth_factor, activation, 
				  optimizer, loss, metrics):

	# Если при создании автоэнкодера с двумя входами был дополнительно указан 
	# параметр concatenate_axis, то проверить, какое ему присвоили значение 
	# (0 или 1), и на основании этого выбора продолжить формирование автоэнкодера
	#
	# 0 – сложить входные слои
	# 1 – сложить первые скрытые слои после входных слоёв
	#
	# Не надо использовать **, всё равно что-то передастся в эту функцию из
	# главной функции (autoenc_gen.py).

	if concatenate_axis == 0:
		return double_inputs_concatenated_inputs(inputs_dimensions, 
												 output_dimension, first_layer, 
												 scale_factor, depth_factor, 
												 activation, optimizer, loss, metrics)
	elif concatenate_axis == 1:
		return double_inputs_concatenated_hl(inputs_dimensions, output_dimension,
											 first_layer, scale_factor, depth_factor, 
											 activation, optimizer, loss, metrics)
	else:
		raise ValueError("concatenate_axis может быть только 0 или 1. "
						 "Передано: {}".format(concatenate_axis))
