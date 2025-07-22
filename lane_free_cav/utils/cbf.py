from lane_free_cav.utils.cbf import QuadraticProgramCBF
self.cbf = QuadraticProgramCBF(gamma=10.0, safe_distance=1.5)

def _apply_cbf(self, actions):
    for a in self.agents:
        safe_act = self.cbf.filter(self._models[a], actions[a],
                                   [self._models[o] for o in self.agents if o!=a])
        actions[a][:] = safe_act
