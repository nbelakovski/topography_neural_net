def evenly_divisible_shape(orig_shape, factor):
    return [x - x % factor for x in orig_shape]
