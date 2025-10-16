import torch
from loguru import logger


def flexible_kernel(X, Y, X_org, Y_org, sigma, sigma0=0.1, epsilon=1e-08):
    """Flexible kernel calculation as in MMDu."""
    Dxy = Pdist2(X, Y)
    Dxy_org = Pdist2(X_org, Y_org)
    L = 1
    Kxy = (1 - epsilon) * torch.exp(
        -((Dxy / sigma0) ** L) - Dxy_org / sigma
    ) + epsilon * torch.exp(-Dxy_org / sigma)
    return Kxy


def MMD_Diff_Var(Kyy, Kzz, Kxy, Kxz, epsilon=1e-08):
    """Compute the variance of the difference statistic MMDXY - MMDXZ."""
    """Referenced from: https://github.com/eugenium/MMD/blob/master/mmd.py"""
    m = Kxy.shape[0]
    n = Kyy.shape[0]
    r = Kzz.shape[0]

    # Remove diagonal elements
    Kyynd = Kyy - torch.diag(torch.diag(Kyy))
    Kzznd = Kzz - torch.diag(torch.diag(Kzz))

    u_yy = torch.sum(Kyynd) * (1.0 / n)
    u_zz = torch.sum(Kzznd) * (1.0 / r)
    u_xy = torch.sum(Kxy) / (m * n)
    u_xz = torch.sum(Kxz) / (m * r)

    t1 = (1.0 / n**3) * torch.sum(Kyynd.T @ Kyynd) - u_yy**2
    t2 = (1.0 / (n**2 * m)) * torch.sum(Kxy.T @ Kxy) - u_xy**2
    t3 = (1.0 / (n * m**2)) * torch.sum(Kxy @ Kxy.T) - u_xy**2
    t4 = (1.0 / r**3) * torch.sum(Kzznd.T @ Kzznd) - u_zz**2
    t5 = (1.0 / (r * m**2)) * torch.sum(Kxz @ Kxz.T) - u_xz**2
    t6 = (1.0 / (r**2 * m)) * torch.sum(Kxz.T @ Kxz) - u_xz**2
    t7 = (1.0 / (n**2 * m)) * torch.sum(Kyynd @ Kxy.T) - u_yy * u_xy
    t8 = (1.0 / (n * m * r)) * torch.sum(Kxy.T @ Kxz) - u_xz * u_xy
    t9 = (1.0 / (r**2 * m)) * torch.sum(Kzznd @ Kxz.T) - u_zz * u_xz

    if type(epsilon) == torch.Tensor:
        epsilon_tensor = epsilon.clone().detach()
    else:
        epsilon_tensor = torch.tensor(epsilon, device=Kyy.device)
    zeta1 = torch.max(t1 + t2 + t3 + t4 + t5 + t6 - 2 * (t7 + t8 + t9), epsilon_tensor)
    zeta2 = torch.max(
        (1 / m) * torch.sum((Kyynd - Kzznd - Kxy.T - Kxy + Kxz + Kxz.T) ** 2)
        - (u_yy - 2 * u_xy - (u_zz - 2 * u_xz)) ** 2,
        epsilon_tensor,
    )

    data = {
        "t1": t1.item(),
        "t2": t2.item(),
        "t3": t3.item(),
        "t4": t4.item(),
        "t5": t5.item(),
        "t6": t6.item(),
        "t7": t7.item(),
        "t8": t8.item(),
        "t9": t9.item(),
        "zeta1": zeta1.item(),
        "zeta2": zeta2.item(),
    }

    Var = (4 * (m - 2) / m) * zeta1
    Var_z2 = Var + (2.0 / m) * zeta2

    return Var, Var_z2, data


def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist < 0] = 0
    return Pdist


def MMD_batch2(
    Fea,
    len_s,
    Fea_org,
    sigma,
    sigma0=0.1,
    epsilon=10 ** (-10),
    is_var_computed=True,
    use_1sample_U=True,
    coeff_xy=2,
):
    X = Fea[0:len_s, :]
    Y = Fea[len_s:, :]
    L = 1  # generalized Gaussian (if L>1)

    nx = X.shape[0]
    ny = Y.shape[0]
    Dxx = Pdist2(X, X)
    Dyy = torch.zeros(Fea.shape[0] - len_s, 1).to(Dxx.device)
    # Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y).transpose(0, 1)
    Kx = torch.exp(-Dxx / sigma0)
    Ky = torch.exp(-Dyy / sigma0)
    Kxy = torch.exp(-Dxy / sigma0)

    nx = Kx.shape[0]

    is_unbiased = False
    xx = torch.div((torch.sum(Kx)), (nx * nx))
    yy = Ky.reshape(-1)
    xy = torch.div(torch.sum(Kxy, dim=1), (nx))

    mmd2 = xx - 2 * xy + yy
    return mmd2


# MMD for three samples
def MMD_3_Sample_Test(
    ref_fea,
    fea_y,
    fea_z,
    ref_fea_org,
    fea_y_org,
    fea_z_org,
    sigma,
    sigma0,
    epsilon,
    alpha
):
    """Run three-sample test (TST) using deep kernel kernel."""
    X = ref_fea.clone().detach()
    Y = fea_y.clone().detach()
    Z = fea_z.clone().detach()
    X_org = ref_fea_org.clone().detach()
    Y_org = fea_y_org.clone().detach()
    Z_org = fea_z_org.clone().detach()

    Kyy = flexible_kernel(Y, Y, Y_org, Y_org, sigma, sigma0, epsilon)
    Kzz = flexible_kernel(Z, Z, Z_org, Z_org, sigma, sigma0, epsilon)
    Kxy = flexible_kernel(X, Y, X_org, Y_org, sigma, sigma0, epsilon)
    Kxz = flexible_kernel(X, Z, X_org, Z_org, sigma, sigma0, epsilon)

    Kyynd = Kyy - torch.diag(torch.diag(Kyy))
    Kzznd = Kzz - torch.diag(torch.diag(Kzz))

    Diff_Var, _, _ = MMD_Diff_Var(Kyy, Kzz, Kxy, Kxz, epsilon)

    # u_yy = torch.sum(Kyynd) / (Y.shape[0] * (Y.shape[0] - 1))
    # u_zz = torch.sum(Kzznd) / (Z.shape[0] * (Z.shape[0] - 1))
    u_yy = torch.sum(Kyynd) / (Y.shape[0])
    u_zz = torch.sum(Kzznd) / (Z.shape[0])
    u_xy = torch.sum(Kxy) / (X.shape[0] * Y.shape[0])
    u_xz = torch.sum(Kxz) / (X.shape[0] * Z.shape[0])

    t = u_yy - 2 * u_xy - (u_zz - 2 * u_xz)

    # Check for NaN values in t
    if torch.isnan(t).any():
        logger.warning("t contains NaN values")
        return 1, 0.0

    # Ensure Diff_Var is positive and not too small
    if Diff_Var.item() <= 0 or torch.isnan(Diff_Var).any():
        Diff_Var = torch.max(torch.tensor(epsilon), torch.tensor(1e-08))

    # Compute the test statistic safely
    sqrt_diff_var = torch.sqrt(Diff_Var)
    if sqrt_diff_var == 0 or torch.isnan(sqrt_diff_var):
        logger.warning("sqrt(Diff_Var) is zero or NaN")
        return 1, 0.0

    test_stat = -t / sqrt_diff_var

    # Check if test_stat is NaN or inf
    if torch.isnan(test_stat).any() or torch.isinf(test_stat).any():
        logger.warning("Test statistic is NaN or inf")
        return 1, 0.0

    p_value = torch.distributions.Normal(0, 1).cdf(test_stat).item()

    return p_value


def h1_mean_var_gram(
    Kx,
    Ky,
    Kxy,
    is_var_computed,
    use_1sample_U=True,
    is_unbiased=True,
    coeff_xy=2,
    is_yy_zero=False,
    is_xx_zero=False,
):
    """compute value of MMD and std of MMD using kernel matrix."""
    if not is_yy_zero:
        coeff_yy = 1
    else:
        coeff_yy = 0
    if not is_xx_zero:
        coeff_xx = 1
    else:
        coeff_xx = 0
    Kxxy = torch.cat((Kx, Kxy), 1)
    Kyxy = torch.cat((Kxy.transpose(0, 1), Ky), 1)
    Kxyxy = torch.cat((Kxxy, Kyxy), 0)
    nx = Kx.shape[0]
    ny = Ky.shape[0]

    if is_unbiased:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div(
                (torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1))
            )
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx * coeff_xx - coeff_xy * xy + yy * coeff_yy
    else:
        xx = torch.div((torch.sum(Kx)), (nx * nx))
        yy = torch.div((torch.sum(Ky)), (ny * ny))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy)), (nx * ny))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx * coeff_xx - coeff_xy * xy + yy * coeff_yy
    if not is_var_computed:
        return mmd2, None, Kxyxy
    hh = Kx * coeff_xx + Ky * coeff_yy - (Kxy + Kxy.transpose(0, 1)) * coeff_xy / 2
    V1 = torch.dot(hh.sum(1) / ny, hh.sum(1) / ny) / ny
    V2 = (hh).sum() / (nx) / nx
    varEst = 4 * (V1 - V2**2)
    if varEst == 0.0:
        logger.warning("error_var!!" + str(V1))
    return mmd2, varEst, Kxyxy


def MMDu(
    Fea,
    Fea_org,
    len_s,
    sigma,
    sigma0=0.1,
    epsilon=10 ** (-10),
    is_smooth=True,
    is_var_computed=True,
    use_1sample_U=True,
    is_unbiased=True,
    coeff_xy=2,
    is_yy_zero=False,
    is_xx_zero=False,
):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = Fea[0:len_s, :]  # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :]  # fetch the sample 2 (features of deep networks)
    X_org = Fea_org[0:len_s, :]  # fetch the original sample 1
    Y_org = Fea_org[len_s:, :]  # fetch the original sample 2
    L = 1  # generalized Gaussian (if L>1)

    nx = X.shape[0]
    ny = Y.shape[0]
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    K_Ix = torch.eye(nx).cuda()
    K_Iy = torch.eye(ny).cuda()
    if is_smooth:
        Kx = (1 - epsilon) * torch.exp(
            -((Dxx / sigma0) ** L) - Dxx_org / sigma
        ) + epsilon * torch.exp(-Dxx_org / sigma)
        Ky = (1 - epsilon) * torch.exp(
            -((Dyy / sigma0) ** L) - Dyy_org / sigma
        ) + epsilon * torch.exp(-Dyy_org / sigma)
        Kxy = (1 - epsilon) * torch.exp(
            -((Dxy / sigma0) ** L) - Dxy_org / sigma
        ) + epsilon * torch.exp(-Dxy_org / sigma)
    else:
        Kx = torch.exp(-Dxx / sigma0)
        Ky = torch.exp(-Dyy / sigma0)
        Kxy = torch.exp(-Dxy / sigma0)

    return h1_mean_var_gram(
        Kx,
        Ky,
        Kxy,
        is_var_computed,
        use_1sample_U,
        is_unbiased,
        coeff_xy,
        is_yy_zero,
        is_xx_zero,
    )