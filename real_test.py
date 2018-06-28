from autoencoder import autoencoder_generator

# Single input

model = autoencoder_generator(number_of_inputs = 1,
							  inputs_dimensions = 42,
							  first_layer = 100,
							  scale_factor = 2,
							  depth_factor = 4,
							  activation = "relu", # или ["relu"]*((depth_factor*2)+1)
							  optimizer="adam",
							  loss="mean_squared_error",
							  metrics=["mse"])

# Double inputs, concatenating input layers

model = autoencoder_generator(number_of_inputs = 2,
							  inputs_dimensions = [400, 42],
							  output_dimension = 42,
							  concatenate_axis = 0,
							  first_layer = 256,
							  scale_factor = 2,
							  depth_factor = 4,
							  activation = "relu", # или ["relu"]*((depth_factor*2)+1)
							  optimizer="adam",
							  loss="mean_squared_error",
							  metrics=["mse"])

# Double inputs, concatenating hidden layers after inputs

model = autoencoder_generator(number_of_inputs = 2,
							  inputs_dimensions = [400, 42],
							  output_dimension = 42,
							  concatenate_axis = 1,
							  first_layer = [128, 128],
							  scale_factor = 2,
							  depth_factor = 4,
							  activation = "relu", # или [["relu", "relu"], "relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu"]
							  optimizer="adam",
							  loss="mean_squared_error",
							  metrics=["mse"])