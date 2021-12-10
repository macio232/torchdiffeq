#!/usr/bin/env python3
import argparse
import math

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event

torch.set_default_dtype(torch.float64)


def linspace(start, stop, N, endpoint=True):
    if endpoint == 1:
        divisor = N - 1
    else:
        divisor = N
    steps = (1.0 / divisor) * (stop - start)
    return steps[:, None] * torch.arange(N) + start[:, None]


class Ballistics(nn.Module):
    def __init__(self, k=0.25, m=0.2, g=9.81, adjoint=False):
        super().__init__()
        self.k = k
        self.m = m
        self.g = g
        self.t0 = nn.Parameter(torch.tensor([0.0]))
        self.odeint = odeint_adjoint if adjoint else odeint

    @staticmethod
    def dposxdt(t, pos_x, pos_y, alpha, velocity, k, m, g):
        return velocity * torch.cos(alpha) * torch.exp(-k * t / m)

    @staticmethod
    def dposydt(t, pos_x, pos_y, alpha, velocity, k, m, g):
        return (
            1
            / k
            * torch.exp(-k * t / m)
            * (k * velocity * torch.sin(alpha) - g * m * (torch.exp(k * t / m) - 1))
        )

    def trajectory(self, x, t):
        pos_x, pos_y, alpha, velocity, _ = x.unsqueeze(-1).expand(-1, -1, t.shape[-1])
        return (
            pos_x
            - velocity
            * torch.cos(alpha)
            * self.m
            / self.k
            * (torch.exp(-self.k * t / self.m) - 1),
            pos_y
            - self.m
            / self.k ** 2
            * (
                (self.g * self.m + velocity * torch.sin(alpha) * self.k)
                * (torch.exp(-self.k * t / self.m) - 1)
                + self.g * t * self.k
            ),
        )

    def grad(self, t, state):
        pos_x, pos_y, alpha, velocity, event_t = state
        grad = torch.stack(
            (
                self.dposxdt(t, pos_x, pos_y, alpha, velocity, self.k, self.m, self.g),
                self.dposydt(t, pos_x, pos_y, alpha, velocity, self.k, self.m, self.g),
                torch.zeros_like(alpha),
                torch.zeros_like(velocity),
                torch.zeros_like(event_t),
            )
        )
        grad[:, event_t.nonzero()] = 0
        return grad

    def event_fn(self, t, state):
        pos_x, pos_y, alpha, velocity, event_t = state
        return torch.where(
            self.dposydt(t, pos_x, pos_y, alpha, velocity, self.k, self.m, self.g) < 0,
            torch.where(event_t == 0, pos_y, torch.tensor(1.0)),
            torch.tensor(1.0),
        )

    def get_initial_state(self, x):
        return self.t0, x

    def update_satate(self, state, idx, event_t):
        state[-1, idx] = event_t
        return state

    def get_collision_times(self, x):

        event_t, solution = self.get_initial_state(x)

        while not (solution[-1, :] != 0).all():
            event_t, solution = odeint_event(
                self.grad,
                solution,
                event_t,
                event_fn=self.event_fn,
                reverse_time=False,
                atol=1e-8,
                rtol=1e-8,
                odeint_interface=self.odeint,
            )
            solution = self.update_satate(
                solution[-1], self.event_fn(event_t, solution[-1]).argmin(), event_t
            )

        return solution[-1, :]
    
    def get_y(self, x, complete=None):
        event_t, solution = self.get_initial_state(x, complete=complete)

        while not (solution[-1, :] != 0).all():
            event_t, solution = odeint_event(
                self.grad,
                solution,
                event_t,
                event_fn=self.event_fn,
                reverse_time=False,
                atol=1e-8,
                rtol=1e-8,
                odeint_interface=self.odeint,
            )
            solution = self.update_satate(
                solution[-1], self.event_fn(event_t, solution[-1]).argmin(), event_t
            )

        return solution[0, :]
    
    def forward(self, x, complete=None, trajectories=False):
        if trajectories:
            return self.simulate(x, complete=complete)
        else:
            return self.get_y(x, complete=complete)
        

    def simulate(self, x):
        event_times = self.get_collision_times(x)

        tt = linspace(
            torch.zeros(1).expand(event_times.size()),
            event_times,
            int((float(event_times.max()) - 0) * 50),
        )
        positions_x, positions_y = self.trajectory(x, tt)
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
    lambda3 = 30
    x = torch.tensor(
        torch.cat(
            (
                torch.randn(n_samples, 2) * torch.tensor([msd0[1], msd1[1]])
                + torch.tensor([msd0[0], msd1[0]]),
                (torch.rand(n_samples, 1) * (int2[1] - int2[0]) + int2[0])
                * math.pi
                / 180,
                torch.poisson(torch.ones(n_samples, 1) * lambda3),
                torch.zeros(n_samples, 1)
            ),
            dim=-1,
        ),
        requires_grad=True,
    ).T

    system = Ballistics()
    times, positions_x, positions_y, = system.simulate(
        x=x,
    )
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
