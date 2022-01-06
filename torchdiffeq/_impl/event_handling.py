import math
import torch


def find_event(interp_fn, sign0, t0, t1, event_fn, tol):
    t1_org = t1
    with torch.no_grad():

        mask = torch.ones_like(sign0, dtype=torch.bool)

        # Num iterations for the secant method until tolerance is within target.
        nitrs = torch.ceil(torch.log((t1 - t0) / tol) / math.log(2.0))

        for _ in range(nitrs.long()):
            t_mid = (t1 + t0) / 2.0
            y_mid = interp_fn(t_mid)
            sign_mid = torch.sign(event_fn(t_mid, y_mid))
            same_as_sign0 = sign0 == sign_mid
            t0 = torch.where(mask, torch.where(same_as_sign0, t_mid, t0), t0)
            t1 = torch.where(mask, torch.where(same_as_sign0, t1, t_mid), t1)
            mask = torch.logical_and(mask, same_as_sign0)
        y_t1 = interp_fn(t1_org)
        sign_t1 = torch.sign(event_fn(t1_org, y_t1))
        same_as_sign0 = sign0 == sign_t1
        t0 = torch.where(mask, torch.where(same_as_sign0, t1_org, t0), t0)
        t1 = torch.where(mask, torch.where(same_as_sign0, t1, t1_org), t1)
        mask = torch.logical_and(mask, same_as_sign0)

        event_t = (t0 + t1) / 2.0
    return event_t, interp_fn(event_t), ~mask


def combine_event_functions(event_fn, t0, y0):
    """
    We ensure all event functions are initially positive,
    so then we can combine them by taking a min.
    """
    with torch.no_grad():
        initial_signs = torch.sign(event_fn(t0, y0))

    def combined_event_fn(t, y):
        c = event_fn(t, y)
        return torch.min(c * initial_signs, dim=0).values
        # return torch.min(c, dim=0).values

    return combined_event_fn
