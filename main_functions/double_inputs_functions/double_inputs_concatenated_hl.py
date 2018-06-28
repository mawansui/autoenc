from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.utils import plot_model
# custom imports
from ..get_used_activations import get_used_activations
from ..get_layer_sizes import get_layer_sizes

# constants
# TODO: put in separate file for clarity
all_available_activation_functions = ["softmax", "elu", "selu", "softplus", 
									  "softsign", "relu", "tanh", "sigmoid", 
									  "hard_sigmoid", "linear"]

def double_inputs_concatenated_hl(inputs_dimensions, output_dimension, first_layer, 
								  scale_factor, depth_factor, activation, 
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
	# 			   TODO:
	#			   		[["1", "2"], "3*"]
	# 			   		То есть если поставить звездочку, то ФА можно настроить разную
	#			   		для разных входов, а потом зафигачить всё одинаковое.
	# optimizer, loss, metrics – стандартные вещи

	# Сначала надо проверить правильность введенных данных.

	if isinstance(inputs_dimensions, list) != True:
		raise ValueError('В параметр inputs_dimensions передан не список. '
						 'Ваш ввод: {}'.format(inputs_dimensions))

	if isinstance(first_layer, list) != True:
		raise ValueError('В параметр first_layer передан не список. '
						 'Ваш ввод: {}'.format(first_layer))

	# Здесь + 1, потому что имеется ввиду один большой список activation, 
	# количество элементов в подсписке для входных уровней здесь конкретно не 
	# учитывается (учитывается далее)
	if isinstance(activation, list) and len(activation) != ((depth_factor * 2) + 1):
		error_message = ("Размер списка в параметре activation некорректен. "
						 "Должно быть {}, а по факту – {}".format((depth_factor * 2) + 1, len(activation)))
		raise ValueError(error_message)

	if isinstance(activation, list) and isinstance(activation[0], list) != True:
		raise ValueError("Если activations – список, то первое значение этого "
						 "списка тоже должно быть вложенным списком.")

	# Вот здесь учитывается, если размерность подсписка неверная. 
	if isinstance(activation, list) and len(activation[0]) != 2:
		raise ValueError("Если activations – список, то первое значение этого "
						 "списка тоже должно быть вложенным списком и содержать "
						 "в себе 2 значения – по одной ФА на каждый входной слой")

	# Теперь надо обработать функции активации. Замес как всегда простой – если
	# одна строка, то создать массив размерностью (depth_factor * 2) + 1 и одним
	# подсписком.

	used_activations = []

	if isinstance(activation, str) and activation in all_available_activation_functions:
		used_activations = get_used_activations(activation, depth_factor)
	elif isinstance(activation, str):
		raise ValueError("В activation передана не строка, либо этой функции"
						 " активации нет в стандартной библиотеке Keras – "
						 "проверьте написание, возможна опечатка. "
						 'Ваш ввод: {}'.format(activation))

	# Если был передан правильный список с подсписком
	if isinstance(activation, list) and len(activation) == ((depth_factor*2)+1) and len(activation[0]) == 2:
		used_activations = activation
	elif isinstance(activation, list):
		raise ValueError("Проверьте то, что передано в параметр activation. ({})".format(activation))

	# А теперь рассчитаем, какое количество нейронов на скольки уровнях должно быть:
	layer_sizes = get_layer_sizes(sum(first_layer), scale_factor, depth_factor)

	# Создаём два входа в автоэнкодер.
	first_input = Input(shape=(inputs_dimensions[0], ))
	second_input = Input(shape=(inputs_dimensions[1], ))

	# Создаём скрытые слои, которые будут идти после каждого входного слоя
	# TODO: узнать, можно ли Concatenate-ить входные слои
	# UPD: блин, можно! Теперь надо попробовать потестить так. 
	# UPD 2: потестил, вынес в отдельную подфункцию.
	# used_activations[0][n] – из первого подсписка взять n-ный размер
	first_hidden_layer = Dense(first_layer[0], activation=used_activations[0][0])(first_input)
	second_hidden_layer = Dense(first_layer[1], activation=used_activations[0][1])(second_input)

	# Это – первый "настоящий" слой автоэнкодера, на нём first_layer[0] + 
	# first_layer[1] нейронов
	concatenated_hidden_layers = Concatenate(axis=-1)([first_hidden_layer, second_hidden_layer])

	connect_to = concatenated_hidden_layers

	# Однообразные операции создания слоёв автоэнкодера
	for i in range(1, depth_factor):
		hidden_layer = Dense(layer_sizes[i], activation=used_activations[i])(connect_to)
		connect_to = hidden_layer

	# Отдельно создаём bottleneck
	bottleneck_layer = Dense(layer_sizes[depth_factor], activation=used_activations[depth_factor])(connect_to)

	# Создаём декодер
	connect_to = bottleneck_layer

	for i in range(depth_factor+1, (depth_factor*2)):
		hidden_layer = Dense(layer_sizes[i], activation=used_activations[i])(connect_to)
		connect_to = hidden_layer

	# Отдельно создаем последний слой
	last_layer = Dense(sum(first_layer), activation=used_activations[(depth_factor*2)])(connect_to)

	# Создаём выходной слой
	output_layer = Dense(output_dimension)(last_layer)

	# Создаём модель автоэнкодера
	model = Model(inputs=[first_input, second_input], outputs=output_layer)

	# Компилируем модель автоэнкодера
	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

	# Выводим в консоль информацию о модели
	model.summary()

	# Рисуем картинку со схемой нового автоэнкодера – сделать опционально?
	plot_model(model, to_file="autoencoder_scheme_3.png", show_shapes=True)

	# Возвращаем модель автоэнкодера
	return model