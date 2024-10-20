import torch
from typing import Union

class RankFactorizationResult:
    def __init__(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        assert A.ndim == 2 and B.ndim == 2, "Mismatched shape"
        assert A.shape[1] == B.shape[1], "Mismatched shape"
        self.A = A
        self.B = B
        self.rank = A.shape[1]
        self.metrics = kwargs

class RegulaizedL1AugmentedLagrangianMethod:
    def __init__(self, **kwargs):
        self.maxiter_in = kwargs.get("maxiter_in", 100)
        self.maxiter_out = kwargs.get("maxiter_out", 5000)
        self.rho = kwargs.get("rho", 1.05)
        self.max_mu = kwargs.get("max_mu", 1e20)
        self.tol = kwargs.get("tol", 1e-8)

    def decompose(
        self,
        D: torch.Tensor,
        W: Union[torch.Tensor, None] = None,
        r=None,
        lambd: Union[float, None] = None,
        device: str = 'cuda'  # Optional device specification
    ):
        # Move data to the desired device (GPU/CPU)
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        D = D.to(device)
        M = D.clone().detach()  # create a copy of the matrix
        m, n = M.shape
        if W is None:
            W = torch.ones_like(D, device=device)

        if r is None:
            r = int(torch.ceil(0.1 * min(m, n)))
        if lambd is None:
            lambd = 1e-3

        # normalization
        scale = torch.max(torch.abs(M))
        M = M / scale

        # initialization
        mu = 1e-6
        M_norm = torch.norm(M, 'fro')
        tol = self.tol * M_norm

        cW = torch.ones_like(W, device=device) - W  # the complement of W
        E = torch.zeros((m, n), device=device).to(torch.float64)
        U = torch.zeros((m, r), device=device).to(torch.float64)
        V = torch.zeros((r, n), device=device).to(torch.float64)
        Y = torch.zeros((m, n), device=device).to(torch.float64)  # Lagrange multiplier

        # start main outer loop
        niter_out = 0
        while niter_out < self.maxiter_out:
            # print(f"do outer loop: {niter_out}")
            niter_out += 1
            niter_in = 0
            obj_pre = torch.tensor(1e20)
            while niter_in < self.maxiter_in:
                # update U
                temp = (E + Y / mu) @ V.T
                Us, sigma, Udt = torch.linalg.svd(temp, full_matrices=False)
                U = Us @ Udt

                # update V
                temp = U.T @ (E + Y / mu)
                Vs, sigma, Vdt = torch.linalg.svd(temp, full_matrices=False)
                svp = torch.sum(sigma > lambd / mu).item()
                if svp >= 1:
                    sigma = sigma[:svp] - lambd / mu
                else:
                    svp = 1
                    sigma = torch.tensor([0.0], dtype=M.dtype, device=device)
                V = Vs[:, :svp] @ torch.diag(sigma.to(Vs.dtype)) @ Vdt[:svp, :]
                sigma0 = sigma
                UV = U @ V

                # update E
                temp1 = UV - Y / mu
                temp = M - temp1
                # E = torch.maximum(temp - 1 / mu, torch.tensor(0., device=device)) + torch.minimum(temp + 1 / mu, torch.tensor(0., device=device))
                E = torch.maximum(temp - 1 / mu, torch.zeros_like(temp)) + torch.minimum(
                    temp + 1 / mu, torch.zeros_like(temp)
                )
                E = (M - E) * W + temp1 * cW
                # evaluate current objective
                obj_cur = (
                    torch.sum(torch.abs(W * (M - E)))
                    + lambd * torch.sum(sigma0)
                    + torch.sum(torch.abs(Y * (E - UV)))
                    + mu / 2 * torch.norm(E - UV, "fro") ** 2
                )
                # check convergence of inner loop
                if torch.abs(obj_cur - obj_pre) < 1e-8 * torch.abs(obj_pre):
                    # print(f"inner loop break at: {niter_in}")
                    break
                else:
                    obj_pre = obj_cur
                    niter_in += 1
            leq = E - UV
            stop_c = torch.norm(leq, "fro")
            # print(stop_c)
            # print(tol)
            # stop_c.shape()
            if stop_c < tol:
                break
            else:
                Y = Y + mu * leq
                mu = min(mu * self.rho, self.max_mu)
        # denormalization
        U_est = torch.sqrt(scale) * U
        V_est = torch.sqrt(scale) * V
        M_est = U_est @ V_est
        l1_error = torch.sum(torch.abs(W * (scale * M - M_est)))

        return RankFactorizationResult(
            A=U_est,
            B=V_est.T,
            convergence={
                "niter": niter_out,
                "stop_c": stop_c.item(),
                "l1_error": l1_error.item(),
                "converged": (niter_out < self.maxiter_out),
            },
        )
