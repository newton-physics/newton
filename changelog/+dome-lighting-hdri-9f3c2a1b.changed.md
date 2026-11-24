Evaluate `SensorCamera` sky/ground ambient lighting through a two-row dome map.
Continue using `set_ambient_light()` with finite, non-negative colors; SH
evaluation introduces small floating-point differences from the previous
hemispheric blend. Dome shadows and backgrounds can also use the default map
without an HDRI file.
