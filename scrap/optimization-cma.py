import catenarysurface as sim
import visualization as vis
import constants as const
import simulation as sim
import cma 
from numpy import fabs
import multiprocessing

def fitness( coeffs ):
    fitness = 0.0

    _ , ballspaths, _ = sim.simulation( visualization=False )

    for ballpositions in ballspaths:
        for ballposition in ballpositions:
            fitness +=  fabs( 1.5 - ballposition[0] ) + fabs( 0.5 - ballposition[1] )
    return( fitness )

def printresult( result ):
    print( "[" + str(result[0]), end=" ")
    for f in result[1:]:
        print( ", " + str(f), end=" " )
    print( "]" )

if __name__ == '__main__':
    es = cma.CMAEvolutionStrategy( [0.0] * const.MAXCOEFF, 1.0,  {'maxfevals': 10000 }  ) # { 'popsize': 75 } {'tolfun': 1e-2, 'maxfevals': 10000, 'popsize': 100 } )

    #for e in cma.CMAOptions():
    #    print( str( e ) )

    pool = multiprocessing.Pool()

    while not es.stop():
        solutions = es.ask()
        fitnessvalues = pool.map(fitness, solutions)
        es.tell(solutions, fitnessvalues)
        es.logger.add()
        es.disp()
        
    pool.close()
    es.result_pretty()
    print( es.result.xbest )
    es.plot()

    # the sampling mean is used to prevent lucky solutions that exploit the simulation to win
    rodspaths, ballspaths, ballsradiuses = sim.simulation( visualization=True )

    vis.generategltffiles( "surfacevisualization", rodspaths, ballspaths, ballsradiuses )

    wait = input("Press Enter to continue.")

    
