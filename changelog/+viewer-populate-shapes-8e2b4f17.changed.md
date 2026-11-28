Speed up viewer `set_model()` for scenes with many shapes by reusing render geometry for repeated shape sources and gathering instance data with NumPy; the rendered batches are unchanged.
