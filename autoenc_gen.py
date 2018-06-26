from main_functions.single_input import single_input
from main_functions.double_inputs import double_inputs

def autoencoder_generator(number_of_inputs, inputs_dimensions, first_layer, 
						  scale_factor, depth_factor, activation, optimizer, 
						  loss, metrics, **kwargs):
	# Функция, создающая автоэнкодер по заданным параметрам
	#
	# number_of_inputs – сколько в автоэнкодере входов, 1 или 2
	#
	# inputs_dimensions – размерность входа (число) или входов, если их 2 (списком)
	# 				   // слово dimension специально везде написано целиком, чтобы
	#				   // избежать путаницы со стандартными параметрами нейросетей
	#				   // в Keras.
	#
	# first_layer – число нейронов на первом скрытом слое автоэнкодера
	#
	# scale_factor – фактор масштабируемости; во сколько раз каждый следующий
	# 				 слой будет меньше (энкодер) или больше предыдущего (декодер)
	#
	# depth_factor – степень глубины; сколько слоёв будет до и после bottleneck-а
	#
	# activation – выбранная функция активации; можно задать автоматически, а
	# 			   можно очень точно для каждого слоя свою
	#
	# **kwargs – это нужно для того, чтобы обработать параметр output_dimension. 
	#			 Он не нужен, если number_of_inputs = 1, там всё автоматически
	#			 присваивается – потому что всё и так понятно. 
	#			 Зато он нужен, если number_of_inputs = 2, потому что надо уто-
	#			 чнить, какой вход мы хотим отразить на выходе.
	#		  // Использую **kwargs, чтобы можно было передавать аргумент 
	#		  // output_dimension не позиционно, а в любом месте по названию.

	# Проверка количества входов.

	# Если у автоэнкодера один вход,
	if number_of_inputs == 1:
		# то надо с помощью функции single_input() из другого файла создать
		# модель автоэнкодера и записать её в переменную model;
		# в функцию single_input() передаётся всё, что было передано в главную 
		# функцию (то есть в эту)
		model = single_input(inputs_dimensions, first_layer, 
							 scale_factor, depth_factor, activation, optimizer, 
							 loss, metrics)
	
	# Если у автоэнкодера два входа,
	else:
		# и если в главную функцию (эту) был передан дополнительный параметр
		# output_dimension, обязательный для создания автоэнкодера с двумя входами,
		if 'output_dimension' in kwargs:
			# то вытащить из аргументов главной функции значение, переданное в
			# output_dimension
			output_dimension = kwargs.get('output_dimension')
			# и передать его вместе со всеми остальными значениями во внешнюю 
			# функцию double_inputs(), которая создаст модель автоэнкодера с
			# двумя входами, которую затем запишем в переменную model
			model = double_inputs(inputs_dimensions, output_dimension, first_layer, 
								  scale_factor, depth_factor, activation, 
								  optimizer, loss, metrics)
		
		# Но если в главную функцию не был передан дополнительный обязательный 
		# параметр output_dimension, то надо поднять ошибку – без него модель
		# не создать.
		else:
			raise ValueError("Не указана размерность выхода для автоэнкодера с двумя входами!")

	# Если на предыдущих этапах всё было ок, вернуть модель.
	return model

new_model = autoencoder_generator(number_of_inputs=1, inputs_dimensions=42, first_layer=100, 
						  scale_factor=2, depth_factor=3, activation="relu", optimizer="adam", 
						  loss="mean_squared_error", metrics=["mse"])