Load the public submodules of `newton`, such as `newton.sensors` and `newton.viewer`, lazily on first access to speed up `import newton`; access them through `newton` or import them as before.
