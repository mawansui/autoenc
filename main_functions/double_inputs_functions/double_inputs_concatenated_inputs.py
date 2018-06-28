from keras.models import Model
from keras.layers import Input, Dense, Concatenate
# custom imports
from ..get_used_activations import get_used_activations
from ..get_layer_sizes import get_layer_sizes

# constants
# TODO: put in separate file for clarity
all_available_activation_functions = ["softmax", "elu", "selu", "softplus", 
									  "softsign", "relu", "tanh", "sigmoid", 
									  "hard_sigmoid", "linear"]

def double_inputs_concatenated_inputs(inputs_dimensions, output_dimension, 
									  first_layer, scale_factor, depth_factor, 
									  activation, optimizer, loss, metrics):
	# Функция создаёт и возвращает модель автоэнкодера с двумя входами и одним выходом.
	# При этом складываются сразу входы, а не первые скрытые слои после них.
	# Это может быть полезно, если данные одинаковые по типу и просто берутся из
	# разных источников. 

	# Проверка правильности введеныных данных
	if isinstance(inputs_dimensions, list) != True:
		raise ValueError('В параметр inputs_dimensions передан не список. '
						 'Ваш ввод: {}'.format(inputs_dimensions))

	if isinstance(activation, list) and len(activation) != ((depth_factor * 2) + 1):
		error_message = ("Размер списка в параметре activation некорректен. "
						 "Должно быть {}, а по факту – {}".format((depth_factor * 2) + 1, len(activation)))
		raise ValueError(error_message)

	# Надо разобраться с активациями. Две проверки – либо это строка, 
	# либо это список.

	used_activations = []

	# Если это строка и она есть в массиве всех доступных функций активации,
	# записать её в массив и сделать его размера (depth_factor*2)+1
	if isinstance(activation, str) and activation in all_available_activation_functions:
		used_activations.append(activation)
		used_activations = used_activations * ((depth_factor * 2) + 1)
	elif isinstance(activation, str):
		raise ValueError("В параметр activation передана строка, которой нет "
						 "в списке всех возможных ФА в Keras. Передано: {}".format(activation))

	if isinstance(activation, list) and len(activation) == ((depth_factor * 2) + 1):
		used_activations = activation
	elif isinstance(activation, list):
		raise ValueError("Проверьте то, что передано в параметр activation. "
						 "Вообще сложно сказать, что нужно сделать, чтобы "
						 "увидеть эту ошибку, оставляю этот текст просто так. "
						 "tiiza - sandman. Передано: {}".format(activation))

	# Содаём список величин всех слоёв
	layer_sizes = get_layer_sizes(first_layer, scale_factor, depth_factor)

	# Создаём два входа в автоэнкодер.
	first_input = Input(shape=(inputs_dimensions[0], ))
	second_input = Input(shape=(inputs_dimensions[1], ))

	# Складываем оба входа в один слой.
	concatenated_inputs = Concatenate(axis=-1)([first_input, second_input])

	# Указываем, что следующий слой будет присоединяться к этому конкатенированному
	connect_to = concatenated_inputs

	# Создаём энкодер. Повторяем однотипную операцию.
	for i in range(0, depth_factor):
		hidden_layer = Dense(layer_sizes[i], activation=used_activations[i])(connect_to)
		connect_to = hidden_layer

	# Создаём боттлнек
	bottleneck_layer = Dense(layer_sizes[depth_factor], activation=used_activations[depth_factor])(connect_to)

	# Создаём декодер
	connect_to = bottleneck_layer

	for i in range(depth_factor+1, (depth_factor*2)):
		hidden_layer = Dense(layer_sizes[i], activation=used_activations[i])(connect_to)
		connect_to = hidden_layer

	# Отдельно создаём последний слой
	last_layer = Dense(layer_sizes[depth_factor*2], activation=used_activations[depth_factor*2])(connect_to)

	# Создаём выходной слой
	output_layer = Dense(output_dimension)(last_layer)

	# Создаём модель
	model = Model(inputs=[first_input, second_input], outputs=output_layer)

	# Компилируем модель
	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

	# Выводим в консоль информацию о модели
	model.summary()

	# Возвращаем модель автоэнкодера
	return model



	return "concatinating inputs"