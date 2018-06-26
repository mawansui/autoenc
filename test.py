import pickle
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Activation, Add, Concatenate
from keras.utils import plot_model

# 1. Multiple Inputs (Reference: https://keras.io/layers/merge/)
# 2. Dense Hidden Layers (Reference: https://keras.io/getting-started/functional-api-guide/)
# 3. Multiple Outputs (Мб не надо? Reference: https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras)
# 4. Prediction Quality Assesement (https://habr.com/company/econtenta/blog/303458/)

with open('./data/x_train.pickle', 'rb') as x_train_file, \
	 open('./data/y_train.pickle', 'rb') as y_train_file, \
	 open('./data/x_test.pickle', 'rb') as x_test_file, \
	 open('./data/y_test.pickle', 'rb') as y_test_file:
	x_train = pickle.load(x_train_file)
	y_train = pickle.load(y_train_file)
	x_test = pickle.load(x_test_file)
	y_test = pickle.load(y_test_file)

# print("x_train\n\n")
# print(y_train)



# Multiple Inputs
# fragments_input: подразумевается, что сюда будет передаваться numpy-массив
# 				   фрагментов реакции
# 
# conditions_input: а сюда – numpy-массив условий реакции
# 
# Таким образом, общее число нейронов на первом слое – np.shape(x_train[1]) + np.shape(y_train[1])
fragments_input = Input(shape=(np.shape(x_train[1])))
conditions_input = Input(shape=(np.shape(y_train[1])))

# После каждого входного слоя идёт слой с 64 нейронами - типа скрытые слои
hidden_layer_1 = Dense(128, activation="relu")(fragments_input)
hidden_layer_2 = Dense(128, activation="relu")(conditions_input)

# Потом складываются скрытые слои в один общий скрытый слой.
# Причем складывать можно только слои с одинаковым числом нейронов.
# added_hidden_layers = Add()([hidden_layer_1, hidden_layer_2])
# Лучше использовать Concatenate – значительно уменьшается функция ошибки
# и вообще как-то по архитектуре сети правильнее.
# Ну, я так изначально и хотел короче, Add() что-то я так и не понял, что делает
added_hidden_layers = Concatenate(axis=-1)([hidden_layer_1, hidden_layer_2])

# Потом просто добавляю скрытых слоёв по вкусу
more_hidden_layers = Dense(128, activation="relu")(added_hidden_layers)
more_hidden_layers = Dense(64, activation="relu")(more_hidden_layers)
more_hidden_layers = Dense(128, activation="relu")(more_hidden_layers)
more_hidden_layers = Dense(256, activation="relu")(more_hidden_layers)

# Добавляется выходной слой
# Вот тут чисто для условий (42) нейрона, или прям всё должно восстанавливаться 
# вместе с фрагментами? 
output_layer = Dense(42)(more_hidden_layers)

# Создаем модель с заданными выше слоями
model = Model(inputs=[fragments_input, conditions_input], outputs=output_layer)

# Распечатываем общую инфу о модели и картинку рисуем
model.summary()
plot_model(model, to_file="modelplot_256_and_128.png", show_shapes=True)
print("hey")

# Вот как-то это всё надо теперь трейнить.

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])

model.fit([x_train, y_train], y_train, batch_size=200, epochs=30, verbose=1, shuffle=False)
print("\nwtf???\n")
predicted_stuff = model.predict([x_test, y_test])
print("Something got predicted.")
print("Length of predicted stuff: {}".format(len(predicted_stuff)))
print("Length of test stuff: {}".format(len(x_test)))

print("First y from test file: {}".format(y_test[0]))
print("\nFirst y from predicted file: {}".format(predicted_stuff[0]))

print("\nSecond y from test file: {}".format(y_test[100]))
print("\nSecond y from predicted file: {}".format(predicted_stuff[100]))

print("\nThird y from test file: {}".format(y_test[2347]))
print("\nThird y from predicted file: {}".format(predicted_stuff[2347]))

# data wrangling to convert first prediction's data from scientific to decimal notation
# first_predicted_y = predicted_stuff[0]
# converted_predicted_y = []

# for number in first_predicted_y:
# 	converted_predicted_y.append(float(number))