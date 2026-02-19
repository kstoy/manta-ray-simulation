from src.physics import catenary as cat


def jet1(x, y, rodheights, D, LF):
    """Compute surface height and gradients at a point."""
    rod_00, rod_10, rod_01, rod_11 = rodheights

    cat_w = cat.findcatenaryparameters(LF, D, rod_00, rod_01)
    cat_e = cat.findcatenaryparameters(LF, D, rod_10, rod_11)
    height_w_y = cat.catenary(y, cat_w)
    height_e_y = cat.catenary(y, cat_e)
    cat_we_x = cat.findcatenaryparameters(LF, D, height_w_y, height_e_y)

    cat_n = cat.findcatenaryparameters(LF, D, rod_01, rod_11)
    cat_s = cat.findcatenaryparameters(LF, D, rod_00, rod_10)
    height_n_x = cat.catenary(x, cat_n)
    height_s_x = cat.catenary(x, cat_s)
    cat_sn_y = cat.findcatenaryparameters(LF, D, height_s_x, height_n_x)

    f = cat.catenary(x, cat_we_x)
    dfx = cat.dcatenary(x, cat_we_x)
    dfy = cat.dcatenary(y, cat_sn_y)

    return [f, dfx, dfy]


def jet1_batch(x, y, rod_00, rod_10, rod_01, rod_11, D, LF):
    """Vectorized jet1: x, y, rod_** are all arrays of length N.

    Returns (f, dfx, dfy) — three arrays of length N.
    """
    # West / East catenary parameters along y
    a_w, ox_w, oy_w = cat.findcatenaryparameters_batch(LF, D, rod_00, rod_01)
    a_e, ox_e, oy_e = cat.findcatenaryparameters_batch(LF, D, rod_10, rod_11)
    height_w_y = cat.catenary_batch(y, a_w, ox_w, oy_w)
    height_e_y = cat.catenary_batch(y, a_e, ox_e, oy_e)
    a_we, ox_we, oy_we = cat.findcatenaryparameters_batch(LF, D, height_w_y, height_e_y)

    # North / South catenary parameters along x
    a_n, ox_n, oy_n = cat.findcatenaryparameters_batch(LF, D, rod_01, rod_11)
    a_s, ox_s, oy_s = cat.findcatenaryparameters_batch(LF, D, rod_00, rod_10)
    height_n_x = cat.catenary_batch(x, a_n, ox_n, oy_n)
    height_s_x = cat.catenary_batch(x, a_s, ox_s, oy_s)
    a_sn, ox_sn, oy_sn = cat.findcatenaryparameters_batch(LF, D, height_s_x, height_n_x)

    f = cat.catenary_batch(x, a_we, ox_we, oy_we)
    dfx = cat.dcatenary_batch(x, a_we, ox_we, oy_we)
    dfy = cat.dcatenary_batch(y, a_sn, ox_sn, oy_sn)

    return f, dfx, dfy
