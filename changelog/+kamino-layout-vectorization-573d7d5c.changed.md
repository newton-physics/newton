Raise `OverflowError` for dense layouts exceeding the `int32` range; catch it instead of `RuntimeError` for DVI or `ValueError` for dense Delassus buffers.
