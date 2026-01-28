from scipy.optimize import differential_evolution, OptimizeResult
from numpy import fabs

import visualization as vis
from config import SimConfig
import simulation as sim

config = SimConfig()


def fitness(cosinewavecoefficients):
    fitness_val = 0.0
    _, ballspaths, _ = sim.simulation(config=config, visualization=False)

    for ballpositions in ballspaths:
        for ballposition in ballpositions:
            fitness_val += fabs(float(config.GRIDSIZEX) - 1.5 - ballposition[0])
    return fitness_val


def printresult(result):
    coeffs = ", ".join(str(f) for f in result.x)
    print(f"[{coeffs}] fitness: {result.fun}")


def thecallback(intermediate_result: OptimizeResult):
    printresult(intermediate_result)
    if intermediate_result.fun < 0.0001:
        raise StopIteration


if __name__ == '__main__':
    bounds = [(-25, 25)] * config.MAXCOEFF

    result = differential_evolution(
        fitness, bounds, workers=10, polish=False,
        updating='deferred', callback=thecallback, tol=0.01
    )

    print("Final result:")
    printresult(result)

    rodspaths, ballspaths, ballsradiuses = sim.simulation(config=config, visualization=True)
    vis.generategltffiles("surfacevisualization", rodspaths, ballspaths, ballsradiuses, config)
