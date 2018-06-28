def get_used_activations(activation, depth):
	# Принимает строку с названием функции активации и степень глубины 
	# автоэнкодера. Возвращает список всех использованных функций активации

	inside_list = []
	outside_list = []

	inside_list.append(activation)

	inside_list = inside_list * 2

	outside_list.append(inside_list)

	while len(outside_list) < (depth*2)+1:
		outside_list.append(activation)

	return outside_list