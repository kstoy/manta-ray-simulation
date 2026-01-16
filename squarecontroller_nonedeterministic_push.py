import numpy as np

class squarecontroller:
    def __init__(self, coeffs):
        self.coeffs = coeffs

        #[NE, NW, SW, SE]

        I = np.array( [0,0,0,0] )

        NE = np.array( [1,0,0,0] )
        NW = np.array( [0,1,0,0] )
        SW = np.array( [0,0,1,0] )
        SE = np.array( [0,0,0,1] )

        E = np.array( [1,0,0,1] ) 
        N = np.array( [1,1,0,0] )
        W = np.array( [0,1,1,0] )
        S = np.array( [0,0,1,1] )

        N_SW = np.array( [1,1,1,0] )
        N_SE = np.array( [1,1,0,1] )
        S_NE = np.array( [1,0,1,1] )
        S_NW = np.array( [0,1,1,1] ) 

        A = np.array( [1,1,1,1] ) 


        self.controller = np.flip( np.array( [
            [SE, S,  S,  S,  S,  SW, SW, SW, SW, SW],
            [SE, S,  S,  S,  S,  SW, W,  W,  W,  W],
            [SE, S,  S,  S,  S,  SW, W,  W,  W,  W],
            [SE, S,  S,  S,  S,  SW, W,  W,  W,  W],
            [SE, SE, SE, SE, I,  I,  W,  W,  W,  W],
            [E,  E,  E,  E,  I,  I,  NW, NW, NW, NW],
            [E,  E,  E,  E,  NE, N,  N,  N,  N,  NW],
            [E,  E,  E,  E,  NE, N,  N,  N,  N,  NW],
            [E,  E,  E,  E,  NE, N,  N,  N,  N,  NW],
            [NE, NE, NE, NE, NE, N,  N,  N,  N,  NW],
        ] ), 0 )

    def update( self, i, j, timestep, sensors ):
        if np.logical_and( self.controller[j][i], sensors ).any():
            return( 1.5 )
        else:
            return( 0.5 )
 

