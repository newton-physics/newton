Fix `ArticulationView` getters for indexed selections of arrays with gradients, which ran their gather kernel on the current default device instead of the model's device.
