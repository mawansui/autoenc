import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Activation, Add, Concatenate
# custom stuff imports
from .max_depth_factor import max_depth_factor
from .get_layer_sizes import get_layer_sizes

# constants
all_available_activation_functions = ["softmax", "elu", "selu", "softplus", 
									  "softsign", "relu", "tanh", "sigmoid", 
									  "hard_sigmoid", "linear"]

def single_input(input_dimension, first_layer, scale_factor, depth_factor, 
				 activation, optimizer, loss, metrics):
	# Функция создает и возвращает модель автоэнкодера с одним входом
	# Смысл у всех аргументов такой же, как и в главной функции (autoenc_gen.py)

	# Первая проверка – если функции активации передали как список, его размер
	# должен совпадать с количеством вообще в целом необходимых для автоэнкодера
	# функций активации ((depth_facor * 2) + 1). Умножение на 2 для того, чтобы 
	# учесть слои энкодера и декодера. + 1 – учитываем то, что после последнего 
	# слоя есть ещё output-слой, перед которым тоже нужен слой активации.
	# Если размерность верная, то всё ок, если нет – выдаст ошибку.

	if isinstance(activation, list) and len(activation) != ((depth_factor * 2) + 1):
		error_message = ("Переданный список функций активации не совпадает по "
						 "размеру с числом скрытых слоёв! Должно быть: {}, по "
						 "факту: {}".format((depth_factor*2)+1, len(activation)))
		raise ValueError(error_message)

	# Потом надо проверить, не была ли функция активации (ФА) передана в виде 
	# строки, и есть ли эта ФА в списке доступных в библиотеке Keras.
	# Если передана одна строка, её для удобства надо превратить в список с 
	# размерностью такой же, сколько скрытых слоёв в автоэнкодере.

	used_activations = []

	if isinstance(activation, str) and activation in all_available_activation_functions:
		used_activations.append(activation)
		used_activations = used_activations * ((depth_factor * 2) + 1)

	# Если же это список с нужным размером (уже проверили сверху), 
	# то просто переназначить его.
	if isinstance(activation, list):
		used_activations = activation

	# Следующая проверка – каким может быть максимально возможный depth_factor
	# для указанных значений first_layer и scale_factor?

	maximum_possible_depth_factor = max_depth_factor(first_layer, scale_factor)

	# Если depth_factor, указанный пользователем, превышает максимально возможный
	# при его параметрах first_layer и scale_factor, поднять поясняющую ошибку
	if depth_factor > maximum_possible_depth_factor:
		df_message_error = ("Указанный depth_factor ({}) не подходит, т.к. "
							"максимально возможный фактор глубины – {}."
							"Предлагаю увеличить количество нейронов на первом "
							"слое автоэнкодера (параметр first_layer), либо "
							"изменить фактор масштабируемости (параметр "
							"scale_factor). Ну или просто уменьшить степень "
							"глубины автоэнкодера.".format(depth_factor, maximum_possible_depth_factor))
		raise ValueError(df_message_error)

	# Создаем входной слой. Его размерность равна размерности передаваемых данных.
	input_layer = Input(shape=(input_dimension, ))

	# Создаем энкодер.

	# connect_to – это переменная, в которой содержится информация о последнем 
	# 			   слое, к которому нужно подключать следующий.
	# Подробнее здесь – https://keras.io/getting-started/functional-api-guide/
	connect_to = input_layer

	# Внимание!

	# Сначала был предложен подход постепенного уменьшения и увеличения размера слоев.
	# Этот подход оказался несправделивым для разного типа задач – нет уникальной формулы,
	# ну я особо и не парился её подбирать (экономия времени).
	# Теперь предлагаю второй вариант: в одном месте посчитать список всех размеров слоёв,
	# и размеры потом пихать в слои по индексам из этого списка. Будет проще, быстрее, и без ошибок.

	# layer_sizes – это список величин слоёв. Пользоваться нужно им.
	layer_sizes = get_layer_sizes(first_layer, scale_factor, depth_factor)

	# Несколько раз выполнить однотипную операцию создания нового слоя с заданным
	# количеством нейронов (изменяющимся в зависимости от степени глубины и 
	# масштаба автоэнкодера) и заданной функцией активации.

	for i in range(0, depth_factor):
		hidden_layer = Dense(layer_sizes[i], activation=used_activations[i])(connect_to)
		connect_to = hidden_layer

	# Отдельно вслед за этим прописываем bottleneck
	bottleneck_layer = Dense(layer_sizes[depth_factor], activation=used_activations[depth_factor])(connect_to)

	# Создаём декодер.
	connect_to = bottleneck_layer

	for i in range(depth_factor+1, (depth_factor*2)):
		hidden_layer = Dense(layer_sizes[i], activation=used_activations[i])(connect_to)
		connect_to = hidden_layer

	# Отдельно создаём последний слой
	last_layer = Dense(first_layer, activation=used_activations[(depth_factor*2)])(connect_to)

	# Наконец, создаём выходной слой, который по размеру такой же, как входные данные
	output_layer = Dense(input_dimension)(last_layer)

	# Создаём модель автоэнкодера
	model = Model(inputs=input_layer, outputs=output_layer)

	# Компилируем модель автоэнкодера
	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

	# Рисуем картинку со схемой нового автоэнкодера – сделать опционально?
	# plot_model(model, to_file="autoencoder_scheme.png", show_shapes=True)

	# Выводим в консоль информацию о модели
	model.summary()

	# Наконец, возвращаем модель
	return model