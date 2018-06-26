def get_layer_sizes(first_layer, scale, depth):
	# Функция принимает размер первого слоя, степень глубины и фактор 
	# масштабируемости и возвращает массив размеров слоёв

	# Этот массив вернется функцией
	layers_sizes = []

	# Первая часть – создание слоёв до bottleneck-а
	# С каждой итерацией слои уменьшаются в scale раз
	for i in range(0, depth+1):
		layers_sizes.append(int(first_layer))
		first_layer = first_layer/scale

	# Вторая часть – просто отразить массив по центру, ведь это автоэнкодер
	for index, element in enumerate(list(reversed(layers_sizes))):
		if index != 0:
			layers_sizes.append(element)

	return layers_sizes