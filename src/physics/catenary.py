from numpy import sinh, cosh, arctanh, sqrt, fabs
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
