description and the docs are coming soon!

for single input autoencoder:
+ number_of_inputs
+ inputs_dimensions
+ first_layer
+ scale_factor
+ depth_factor
+ activation
+ optimizer
+ loss
+ metrics

for two input autoencoder concatenating hidden layers:
+ number_of_inputs
+ inputs_dimensions (list)
+ output_dimension
+ concatenate_axis (optional, default for concat_hl == 1)
+ first_layer (list)
+ scale_factor
+ depth_factor
+ activation (string or list with a sublist)
+ optimizer
+ loss
+ metrics


for two input autoencoder concatenating input layers:
+ number_of_inputs
+ inputs_dimensions (list)
+ output_dimension
+ concatenate_axis = 0

The following is ambiguous.
+ first_layer (list)
+ scale_factor
+ depth_factor
+ activation (string or list with a sublist)
+ optimizer
+ loss
+ metrics


Global TODOs:
- [ ] Move activation functions list to separate file
- [ ] Add more required error checkers and error raisers!
- [ ] Move all error messages to one file and assign them with a number
- [ ] Make clear why I always create last levels separately from decoder levers
