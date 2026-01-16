import numpy as np
import time

import constants as const

import visualization as vis
import simulation as sim 

def runexperiment():
    for ballcount in range( 0, 105, 5 ):
        const.NBALL = ballcount

        times = []
        for count in range(10):
            start = time.time()
            rodsstates, ballsstates, ballsradiuses = sim.simulation( [ 0.29266049,  0.85611358, -2.24317114,  1.36182964, 0.4858719 ], visualization=False )
            end = time.time()
            times.append( end - start )

        print(str(ballcount) + " " + np.average( times ).astype(str) + " " + np.std( times ).astype(str) )
    

    print("\n\n")

if __name__ == "__main__":

    print("# plot \"timevsballs.dat\" index 0 title \"5x5 surface\" with errorlines, \"timevsballs.dat\" index 1 title \"10x10 surface\" with errorlines")
    print("# 5x5, 100 timesteps, uniform")

    const.GRIDSIZEX = 5           
    const.GRIDSIZEY = 5
    runexperiment()

    print("# 10x10, 100 timesteps, uniform")

    const.GRIDSIZEX = 10           
    const.GRIDSIZEY = 10
    runexperiment()
