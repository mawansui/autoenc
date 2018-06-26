def max_depth_factor(initial_layer_size, scale_factor):
	# Функция подсчитывает максимально возможный фактор глубины для заданных
	# значений величины первого (исходного) слоя и фактора масштабируемости
	x = initial_layer_size
	counter = 0
	while x > 1:
		x = x/scale_factor
		counter += 1
	# Потому что последняя операция (x/2) приводит к результату (0; 1), а число 
	# нейронов не может быть меньше единицы.
	return counter-1

# print(max_depth_factor(256, 4))