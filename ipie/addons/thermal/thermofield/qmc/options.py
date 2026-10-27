# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#

from dataclasses import dataclass
from typing import Optional, Tuple

from ipie.addons.thermal.qmc.options import ThermalQMCParams


@dataclass
class ThermofieldQMCParams(ThermalQMCParams):
    r"""Input options and certain constants / parameters derived from them.

    The fields of :class:`ThermalQMCParams`, plus:

    Attributes
    ----------
    eta_members : tuple
        Trial times of the mixed estimators. Every eta / timestep must be whole
        and 0 <= eta < beta. Default (beta / 2, beta / 4, beta / 8).
    measure_trace : bool
        Measure the closed trace at the end of the walk. With False the walk
        stops at the last member, beta - min(eta). Default True.
    weight_cap : float
        Walker weights are clipped at this fraction of the total weight in the
        phaseless walk; 0 disables the cap. Default 0.10.
    """

    eta_members: Optional[Tuple[float, ...]] = None
    measure_trace: bool = True
    weight_cap: float = 0.10

    def __post_init__(self):
        super().__post_init__()
        if self.eta_members is None:
            self.eta_members = (0.5 * self.beta, 0.25 * self.beta, 0.125 * self.beta)
        self.eta_members = tuple(sorted((float(eta) for eta in self.eta_members), reverse=True))
        if not self.measure_trace and len(self.eta_members) == 0:
            raise ValueError("Nothing to measure: no eta members and measure_trace=False.")
