Fix `ModelBuilder.add_shape_plane` for a plane equation whose normal points along -Z (it produced a plane facing +Z) and for plane equations given as integers (it raised a casting error).
