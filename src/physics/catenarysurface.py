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
