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

    Kyy.fill_diagonal_(0.0)
    Kzz.fill_diagonal_(0.0)
    Kyynd, Kzznd = Kyy, Kzz

    u_yy = torch.sum(Kyynd) * (1.0 / n)
    u_zz = torch.sum(Kzznd) * (1.0 / r)
    # use .mean() which fuses the reduction & division on CUDA for speed
    u_xy = Kxy.mean()
    u_xz = Kxz.mean()

    # use einsum reductions to avoid explicit matrix products
    t1 = (1.0 / n**3) * torch.einsum('ij,ij->', Kyynd, Kyynd) - u_yy**2
    t2 = (1.0 / (n**2 * m)) * torch.einsum('ij,ij->', Kxy, Kxy) - u_xy**2
    t3 = (1.0 / (n * m**2)) * torch.einsum('ij,ij->', Kxy, Kxy) - u_xy**2  # symmetric reuse
    t4 = (1.0 / r**3) * torch.einsum('ij,ij->', Kzznd, Kzznd) - u_zz**2
    t5 = (1.0 / (r * m**2)) * torch.einsum('ij,ij->', Kxz, Kxz) - u_xz**2
    t6 = (1.0 / (r**2 * m)) * torch.einsum('ij,ij->', Kxz, Kxz) - u_xz**2
    t7 = (1.0 / (n**2 * m)) * torch.einsum('ij,ij->', Kyynd, Kxy) - u_yy * u_xy
    t8 = (1.0 / (n * m * r)) * torch.einsum('ij,ij->', Kxy, Kxz) - u_xz * u_xy
    t9 = (1.0 / (r**2 * m)) * torch.einsum('ij,ij->', Kzznd, Kxz) - u_zz * u_xz

    if isinstance(epsilon, torch.Tensor):
        epsilon_tensor = epsilon.detach().clone().to(Kyy.device)
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

    Var = (4 / m) * zeta1
    Var_z2 = Var + (2.0 / m) * zeta2

    return Var, Var_z2, data


def Pdist2(x, y):
    """compute the paired distance between x and y."""
    if y is None:
        y = x
    Pdist = torch.cdist(x, y, p=2).pow(2)
    return Pdist


def MMD_3_Sample_Test(
    ref_fea,
    fea_y,
    fea_z,
    ref_fea_org,
    fea_y_org,
    fea_z_org,
    sigma,
    sigma0,
    epsilon
):
    """Run three-sample test (TST) using deep kernel kernel."""
    Kyy = flexible_kernel(fea_y, fea_y, fea_y_org, fea_y_org, sigma, sigma0, epsilon)
    Kzz = flexible_kernel(fea_z, fea_z, fea_z_org, fea_z_org, sigma, sigma0, epsilon)
    Kxy = flexible_kernel(ref_fea, fea_y, ref_fea_org, fea_y_org, sigma, sigma0, epsilon)
    Kxz = flexible_kernel(ref_fea, fea_z, ref_fea_org, fea_z_org, sigma, sigma0, epsilon)

    u_yy = torch.sum(Kyy) / (fea_y.shape[0])
    u_zz = torch.sum(Kzz) / (fea_z.shape[0])
    u_xy = Kxy.mean()
    u_xz = Kxz.mean()

    t = u_yy - 2 * u_xy - (u_zz - 2 * u_xz)

    Diff_Var, _, _ = MMD_Diff_Var(Kyy, Kzz, Kxy, Kxz, epsilon)

    if Diff_Var.item() <= 1e-8 or torch.isnan(Diff_Var).any():
        logger.warning(f"Diff_Var is too small, zero, negative or NaN. Diff_Var: {Diff_Var.item()}")
        Diff_Var = torch.tensor(max(epsilon.item() if isinstance(epsilon, torch.Tensor) else epsilon, 1e-08), device=Diff_Var.device)

    sqrt_diff_var = torch.sqrt(Diff_Var)
    test_stat = -t / sqrt_diff_var

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

    # 1) compute pairwise squared distances once for merged tensors
    merged = torch.cat([X, Y], dim=0)
    merged_org = torch.cat([X_org, Y_org], dim=0)

    with torch.amp.autocast("cuda", enabled=merged.is_cuda):
        D_all = Pdist2(merged, merged)           # (N,N)
        D_all_org = Pdist2(merged_org, merged_org)

        Dxx = D_all[:nx, :nx]
        Dyy = D_all[nx:, nx:]
        Dxy = D_all[:nx, nx:]

        Dxx_org = D_all_org[:nx, :nx]
        Dyy_org = D_all_org[nx:, nx:]
        Dxy_org = D_all_org[:nx, nx:]

        # 2) fused kernel helper
        inv_sigma = 1.0 / sigma
        inv_sigma0 = 1.0 / sigma0
        one_minus_ep = 1.0 - epsilon

        def _kernel(dist_hidden, dist_org):
            if is_smooth:
                return one_minus_ep * torch.exp(-((dist_hidden * inv_sigma0) ** L) - dist_org * inv_sigma) \
                       + epsilon * torch.exp(-dist_org * inv_sigma)
            else:
                return torch.exp(-dist_hidden * inv_sigma0)

        Kx = _kernel(Dxx, Dxx_org)
        Ky = _kernel(Dyy, Dyy_org)
        Kxy = _kernel(Dxy, Dxy_org)

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