from numpy import sinh, cosh, arctanh, sqrt, fabs
import numpy as np
import sys


def catenary(x, parameters):
    """The catenary function."""
    a, offsetx, offsety = parameters
    return a * cosh((x + offsetx) / a) + offsety


def dcatenary(x, parameters):
    """Differentiated catenary function."""
    a, offsetx, offsety = parameters
    return sinh((x + offsetx) / a)


def findcatenaryparameters(l, d, h1, h2):
    """
    Find the parameters of a catenary function.

    Based on the paper "Catenary Curve" by Rod Deakin:
    https://www.mygeodesy.id.au/documents/Catenary%20Curve.pdf

    Args:
        l: length of chain
        d: distance between the attachments of the chain
        h1, h2: heights of the attachments

    Returns:
        [a, offsetx, offsety] parameters for the catenary curve
    """
    v = h2 - h1
    d_straight = sqrt(v**2 + d**2)

    if d_straight > l:
        print("Chain too short! Minimum required length: " + str(sqrt(fabs(v)**2 + d**2)))
        print(f"{l} {d} {h1} {h2}")
        sys.exit(1)

    # Estimate a (equation 42 of Deakin's paper)
    a = d / sqrt(24) * sqrt(d / (sqrt(l**2 - v**2) - d))

    # Translate the function to match the known end points
    x1 = a * arctanh(v / l) - d / 2
    offsetx = x1
    y1 = a * cosh(x1 / a)
    offsety = h1 - y1

    return [a, offsetx, offsety]


# --------------- Batch (vectorized) versions ---------------

def findcatenaryparameters_batch(l, d, h1, h2):
    """Vectorized version: h1, h2 are arrays of length N.

    Returns (a, offsetx, offsety) — three arrays of length N.
    """
    v = h2 - h1
    a = d / np.sqrt(24.0) * np.sqrt(d / (np.sqrt(l * l - v * v) - d))
    x1 = a * np.arctanh(v / l) - d / 2.0
    y1 = a * np.cosh(x1 / a)
    offsety = h1 - y1
    return a, x1, offsety


def catenary_batch(x, a, offsetx, offsety):
    """Vectorized catenary: all args are arrays of length N (or broadcastable)."""
    return a * np.cosh((x + offsetx) / a) + offsety


def dcatenary_batch(x, a, offsetx, offsety):
    """Vectorized catenary derivative."""
    return np.sinh((x + offsetx) / a)
