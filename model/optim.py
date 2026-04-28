import torch

def _zeropower_newton_schulz(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    original_shape = matrix.shape
    if matrix.ndim != 2:
        matrix = matrix.reshape(matrix.shape[0], -1)

    work = matrix.float()
    if work.shape[0] > work.shape[1]:
        work = work.T
        transposed = True
    else:
        transposed = False

    work = work / (work.norm() + eps)
    for _ in range(steps):
        a = work @ work.T
        b = 0.5 * (3.0 * torch.eye(a.shape[0], device=work.device, dtype=work.dtype) - a)
        work = b @ work

    if transposed:
        work = work.T
    return work.reshape(original_shape).to(matrix.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            beta1, beta2 = group["adam_betas"]
            adam_eps = group["adam_eps"]

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[param]
                if param.ndim >= 2:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(param)

                    update = grad
                    if weight_decay != 0:
                        update = update.add(param, alpha=weight_decay)

                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(update)
                    if nesterov:
                        update = update.add(buf, alpha=momentum)
                    else:
                        update = buf

                    update = _zeropower_newton_schulz(update, steps=ns_steps)
                    param.add_(update, alpha=-lr)
                    continue

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(adam_eps)
                step_size = lr / bias_correction1
                param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
