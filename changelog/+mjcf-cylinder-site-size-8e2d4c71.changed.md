Import MJCF cylinder sites with `scale[2]` set to `0.0` instead of passing through the third `size` component, which MuJoCo ignores for cylinders and Newton would interpret as a barrel radius.
