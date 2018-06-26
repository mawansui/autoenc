def double_inputs(inputs_dimensions, output_dimension, 
				  first_layer, scale_factor, depth_factor, activation, 
				  optimizer, loss, metrics):
	# Функция создает и возвращает модель автоэнкодера с двумя входами и одним выходом
	# Смысл у всех аргументов такой же, как и в главной функции (autoenc_gen.py).
	#
	# inputs_dimensions – список
	# output_dimension – одно число
	# first_layer – список из двух чисел
	# scale_factor – одно число
	# depth_factor – одно число
	# activation – одна строка (везде эта ФА), либо список со вложенным списком
	# 			   [ ["1", "2"], "3", "4", ..., "N" ] 
	# 			   То, то во вложенном списке – это ФА для первых слоёв на 
	# 			   каждом входе, первая – для первого, вторая – для второго. 
	# 			   Все остальные – для всех остальных последующих слоёв.
	# 			   А ещё можно сделать так, что
	#			   [["1", "2"], "3*"]
	# 			   То есть если поставить звездочку, то ФА можно настроить разную
	#			   для разных входов, а потом зафигать всё одинаковое.
	# optimizer, loss, metrics – стандартные вещи

	# Сначала надо проверить правильность введенных данных.

	if isinstance(inputs_dimensions, list) != True:
		raise ValueError('В параметр inputs_dimensions передан не список.')

	if isinstance(first_layer, list) != True:
		raise ValueError('В параметр first_layer передан не список.')

	if isinstance(activation, list) and len(activation) != ((depth_factor * 2) + 1):
		error_message = ("Размер списка в параметре activation некорректен. "
						 "Должно быть {}, а по факту – {}".format((depth_factor * 2) + 1), len(activation))
		raise ValueError(error_message)

	if isinstance(activation, list) and isinstance(activation[0], list) != True:
		raise ValueError("Если activations – список, то первое значение этого "
						 "списка тоже должно быть вложенным списком.")

	if isinstance(activation, list) and len(activation[0]) != 2:
		raise ValueError("Если activations – список, то первое значение этого "
						 "списка тоже должно быть вложенным списком и содержать "
						 "в себе 2 значения – по одной ФА на каждый входной слой")

	return "hey"
