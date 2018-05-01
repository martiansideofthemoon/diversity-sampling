"""These macros allow adaptive loss modes."""


def l1(epoch, batch, args=None):
    """Stick to l1 loss"""
    return 1.0


def l2(epoch, batch, args=None):
    """Stick to l2 loss"""
    return 0.0


def alternate(epoch, batch, args=None):
    if batch % 2 == 0:
        return 1.0
    else:
        return 0.0


def mixed(epoch, batch, args=None):
    return args.mixed_constant
