#!/usr/bin/env python3
from types import MappingProxyType
from typing import Literal

import argparse
import math

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn

from torchdiffeq import odeint
from torchdiffeq import odeint_event

torch.set_default_dtype(torch.float32)


def linspace(start, stop, N, endpoint=True):
    if endpoint == 1:
        divisor = N - 1
    else:
        divisor = N
    steps = (1.0 / divisor) * (stop - start)
    return steps[:, None] * torch.arange(N) + start[:, None]


class Ballistics(nn.Module):
    def __init__(
        self,
        k=0.25,
        m=0.2,
        g=9.81,
        method="euler",
        method_options=MappingProxyType({"step_size": 0.01}),
        event_options=MappingProxyType({}),
        y_type: Literal["implicit", "explicit"] = "implicit",
    ):
        super().__init__()
        self.y_type = y_type
        self.k = k
        self.m = m
        self.g = g
        self.t0 = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.odeint = odeint
        self.method = method
        self.event_options = event_options
        if "dtype" in method_options:
            self.method_options = dict(**method_options)
            self.method_options["dtype"] = getattr(
                torch, self.method_options["dtype"]
            )
        else:
            self.method_options = method_options

    def dposxdt(self, t, pos_x, pos_y, alpha, velocity, complete=None):
        return velocity * torch.cos(alpha) * torch.exp(-self.k * t / self.m)

    def dposydt(self, t, pos_x, pos_y, alpha, velocity, complete=None):
        return (
            1
            / self.k
            * torch.exp(-self.k * t / self.m)
            * (
                self.k * velocity * torch.sin(alpha)
                - self.g * self.m * (torch.exp(self.k * t / self.m) - 1)
            )
        )

    def trajectory(self, x, t, complete=None, both=True):
        if t.dim() == 1 or t.dim() == 0:
            pos_x, pos_y, alpha, velocity = x
        elif t.dim() == 2:
            pos_x, pos_y, alpha, velocity = x.unsqueeze(-1).expand(
                -1, -1, t.shape[-1]
            )
        else:
            raise ValueError(f"Too many dimensions for time: {t.dim()}")
        velocity = velocity * 20
        if both is True:
            return (
                self._trajectory_x(alpha, pos_x, t, velocity),
                self._trajectory_y(alpha, pos_y, t, velocity),
            )
        elif both == "x":
            return self._trajectory_x(alpha, pos_x, t, velocity)
        elif both == "y":
            return self._trajectory_y(alpha, pos_y, t, velocity)
        else:
            raise ValueError(f"Unknown trajectory setting {both}")

    def _trajectory_y(self, alpha, pos_y, t, velocity):
        return pos_y - self.m / self.k ** 2 * (
            (self.g * self.m + velocity * torch.sin(alpha) * self.k)
            * (torch.exp(-self.k * t / self.m) - 1)
            + self.g * t * self.k
        )

    def _trajectory_x(self, alpha, pos_x, t, velocity):
        return (
            pos_x
            - velocity
            * torch.cos(alpha)
            * self.m
            / self.k
            * (torch.exp(-self.k * t / self.m) - 1)
        ) / 10

    def grad(self, t, state, complete=None):
        pos_x, pos_y, alpha, velocity, event_t = state
        grad = torch.stack(
            (
                self.dposxdt(
                    t,
                    pos_x,
                    pos_y,
                    alpha,
                    velocity,
                    complete=complete,
                ),
                self.dposydt(
                    t,
                    pos_x,
                    pos_y,
                    alpha,
                    velocity,
                    complete=complete,
                ),
                torch.zeros_like(alpha),
                torch.zeros_like(velocity),
                torch.zeros_like(event_t),
            )
        )
        return torch.where(
            event_t == 0, grad, torch.tensor(0.0, device=state.device)
        )

    def event_fn(self, t, state, complete=None):
        pos_x, pos_y, alpha, velocity, event_t = state
        return torch.where(
            self.dposydt(
                t,
                pos_x,
                pos_y,
                alpha,
                velocity,
                complete=complete,
            )
            <= 0,
            torch.where(
                event_t == 0, pos_y, torch.tensor(1000.0, device=state.device)
            ),
            torch.tensor(1000.0, device=state.device),
        ).unsqueeze(0)

    def get_initial_state(self, x, complete=None):
        return self.t0, torch.cat(
            (
                x[:3, :],
                x[3:4, :] * 20,
                torch.zeros(
                    1,
                    x.shape[-1],
                    device=x.device,
                    requires_grad=x.requires_grad,
                ),
            ),
            dim=0,
        )

    def update_state(self, state, mask, event_t, complete=None):
        return torch.cat(
            (
                state[:-1, :],
                torch.where(mask, event_t, state[-1, :]).unsqueeze(0),
            ),
            dim=0,
        )

    def get_initial_t(self, event_t, mask, complete=None):
        return event_t[mask].mean()

    def get_collision_times(self, x, complete=None):
        return self._run_forward(x, complete=complete)[-1, :].T

    def get_y(self, x, complete=None):
        return self._run_forward(x, complete=complete)[0, :].T.unsqueeze(-1)

    def _run_forward(self, x, complete=None):
        t0, solution = self.get_initial_state(x, complete=complete)
        event_t = None
        pbar = tqdm(total=x.shape[-1], disable=True)
        while not (solution[-1, :] != 0).all():
            if event_t is not None:
                t0 = self.get_initial_t(
                    event_t, solution[-1, :] == 0, complete=complete
                )
            event_t, solution, location = odeint_event(
                lambda t, state: self.grad(t, state, complete=complete),
                solution,
                t0,
                event_fn=lambda t, state: self.event_fn(
                    t, state, complete=complete
                ),
                reverse_time=False,
                **self.event_options,
                odeint_interface=self.odeint,
                options=self.method_options,
                method=self.method,
            )
            pbar.update(location.sum().item())
            solution = self.update_state(
                solution[-1],
                location,
                event_t,
                complete=complete,
            )

        pbar.close()
        return torch.cat((solution[:1, :] / 10, solution[1:, :]), dim=0)

    def forward(self, x, complete=None, trajectories=False):
        if trajectories:
            return self.simulate(x.T, complete=complete)
        else:
            if self.y_type == "implicit":
                return self.get_y(x.T, complete=complete)
            elif self.y_type == "explicit":
                return self.trajectory(
                    x.T,
                    self.get_collision_times(x.T, complete=complete),
                    both="x",
                ).unsqueeze(-1)
            else:
                raise ValueError(f"Unknown y computation type: {self.y_type}")

    def simulate(self, x, complete=None):
        event_times = self.get_collision_times(x, complete=complete)

        tt = linspace(
            torch.zeros(1, device=event_times.device).expand(
                event_times.size()
            ),
            event_times,
            int((float(event_times.max()) - 0) * 50),
        )
        positions_x, positions_y = self.trajectory(x, tt, complete=complete)
        return (
            tt,
            positions_x,
            positions_y,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--adjoint", action="store_true")
    args = parser.parse_args()
    n_samples = 16
    msd0 = (0, 0.25)
    msd1 = (1.5, 0.25)
    int2 = (9, 72)
    lambda3 = 15
    x = torch.tensor(
        torch.cat(
            (
                torch.randn(n_samples, 2) * torch.tensor([msd0[1], msd1[1]])
                + torch.tensor([msd0[0], msd1[0]]),
                (torch.rand(n_samples, 1) * (int2[1] - int2[0]) + int2[0])
                * math.pi
                / 180,
                torch.poisson(torch.ones(n_samples, 1) * lambda3) / 20,
            ),
            dim=-1,
        ),
        requires_grad=True,
    )

    system = Ballistics()
    (
        times,
        positions_x,
        positions_y,
    ) = system.forward(x=x, trajectories=True)
    n_plots = n_samples

    positions_x = positions_x.detach().cpu().numpy()
    positions_y = positions_y.detach().cpu().numpy()

    fig, axs = plt.subplots(
        nrows=math.ceil(math.sqrt(n_plots)),
        ncols=math.ceil(math.sqrt(n_plots)),
        figsize=(16, 16),
    )
    axs = axs.ravel()
    for idx, ax in enumerate(axs):
        ax.title.set_text(f"Event t: {times[idx, -1].item():.4f}")
        ax.plot(
            positions_x[idx, -1],
            0.0,
            color="C0",
            marker="o",
            markersize=7,
            fillstyle="none",
            linestyle="",
        )

        pos = ax.plot(
            positions_x[idx, :], positions_y[idx, :], color="C0", linewidth=2.0
        )

        ax.set_xlim([positions_x[idx, 0] - 0.1, positions_x[idx, -1] + 0.1])

        ax.xaxis.set_tick_params(
            direction="in", which="both"
        )  # The bottom will maintain the default of 'out'
        ax.yaxis.set_tick_params(
            direction="in", which="both"
        )  # The bottom will maintain the default of 'out'

        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        if idx == n_plots - 1:
            break

    plt.tight_layout()
    plt.savefig("ballistics.png")
