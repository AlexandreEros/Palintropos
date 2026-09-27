"""Time steppers: *how* one step advances a coefficient array.

The scheduler in :mod:`tropoi.temporal.integration` decides *when* the
solver steps and stores; a stepper owns the arithmetic of one step and the
time levels it carries between steps. The split lets a multi-level scheme
(semi-implicit leapfrog) live behind the same interface as one-level RK4,
and lets a checkpoint serialize exactly the state a scheme needs to resume
bit-identically (:meth:`TimeStepper.state_dict`).

Import-light: NumPy for the state-dict copies; CuPy arrays pass through
untouched because every operation is duck-typed on the array module.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, runtime_checkable

from tropoi.temporal.integration import rk4_step_array


@runtime_checkable
class TimeStepper(Protocol):
    """One-step time integrator over a coefficient array.

    ``initialize(y0, t0)`` sets the starting state; ``step()`` advances by
    ``dt`` and returns the new time; ``state`` is the current state array;
    ``state_dict()`` / ``load_state_dict()`` capture and restore *every* time
    level and counter the scheme needs so that a restored stepper continues
    bit-identically on the same software/hardware configuration.
    """

    dt: float

    def initialize(self, y0, t0: float = 0.0) -> None: ...
    def step(self) -> float: ...
    @property
    def state(self): ...
    @property
    def t(self) -> float: ...
    @property
    def step_count(self) -> int: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, d: dict[str, Any]) -> None: ...


class RK4Stepper:
    """Classical RK4 through :func:`rk4_step_array` (bit-identical to it).

    ``stage_validator`` is forwarded unchanged (called on the three
    intermediate stages).
    """

    def __init__(self, tendency: Callable, dt: float, *,
                 stage_validator: Optional[Callable] = None):
        self.tendency = tendency
        self.dt = float(dt)
        self.stage_validator = stage_validator
        self._y = None
        self._t = 0.0
        self._n = 0

    def initialize(self, y0, t0: float = 0.0) -> None:
        self._y = y0.copy()
        self._t = float(t0)
        self._n = 0

    def step(self) -> float:
        if self._y is None:
            raise RuntimeError("RK4Stepper.step() before initialize()")
        self._y = rk4_step_array(self.tendency, self._y, self._t, self.dt,
                                 stage_validator=self.stage_validator)
        self._t += self.dt
        self._n += 1
        return self._t

    @property
    def state(self):
        return self._y

    @property
    def t(self) -> float:
        return self._t

    @property
    def step_count(self) -> int:
        return self._n

    def state_dict(self) -> dict[str, Any]:
        return {"scheme": "rk4", "dt": self.dt, "t": self._t,
                "step": self._n, "y": self._y.copy()}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        if d.get("scheme") != "rk4":
            raise ValueError(f"state dict is for scheme {d.get('scheme')!r}, not rk4")
        if float(d["dt"]) != self.dt:
            raise ValueError(f"state dict dt={d['dt']} differs from stepper dt={self.dt}")
        self._y = d["y"].copy()
        self._t = float(d["t"])
        self._n = int(d["step"])
