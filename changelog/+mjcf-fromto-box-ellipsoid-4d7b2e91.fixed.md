Fixed `add_mjcf()` raising `IndexError` for `box` and `ellipsoid` geoms that use `fromto`; they now span the segment like MuJoCo.
