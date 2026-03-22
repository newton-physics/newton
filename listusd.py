from pxr import Usd

stage = Usd.Stage.Open("/home/pkorzeniowsk/Projects/newton/newton/newton/examples/cosserat/models/AortaWithVesselsStatic.usdc")

# Traverse all prims in the stage
for prim in stage.Traverse():
    print(f"{prim.GetPath()} ({prim.GetTypeName()})")