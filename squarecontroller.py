import numpy as np
from constants import  *

class squarecontroller:
    def __init__(self, coeffs):
        self.coeffs = coeffs

        self.controller = np.flip( np.array( [
            [E, E, E, E, SE, SW, W, W, W, W],
            [E, E, E, E, SE, SW, W, W, W, W],
            [E, E, E, E, SE, SW, W, W, W, W],
            [E, E, E, E, SE, SW, W, W, W, W],
            [E, E, E, E, I, I, W, W, W, W],
            [E, E, E, E, I, I, W, W, W, W],
            [E, E, E, E, NE, NW, W, W, W, W],
            [E, E, E, E, NE, NW, W, W, W, W],
            [E, E, E, E, NE, NW, W, W, W, W],
            [E, E, E, E, NE, NW, W, W, W, W]
        ] ), 0 )

    def square( self, i, j,sensors ):

        if self.controller[j][i] == 10:
            return 1.5
        if self.controller[j][i] == E:
            if ( sensors[SE] or sensors[NE] ):
                return 1.5
            elif ( sensors[SW] or sensors[NW] ):
                return 0.5
            else:
                return 0.5
        elif self.controller[j][i] == W:
            if ( sensors[SW] or sensors[NW] ):
                return 1.5
            elif ( sensors[SE] or sensors[NE] ):
                return 0.5
            else:
                return 0.5
        elif self.controller[j][i] == N:
            if ( sensors[NW] or sensors[NE] ):
                return 0.5
            elif ( sensors[SE] or sensors[SW] ):
                return 1.5
            else:
                return 0.5
        elif self.controller[j][i] == NW:
            if ( sensors[NW] ):
                return 1.5
            elif ( sensors[SW] ):
                return 0.5            
            else:
                return 0.5
        elif self.controller[j][i] == NE:
            if ( sensors[NE] ):
                return 1.5
            elif ( sensors[SE] ):
                return 0.5            
            else:
                return 0.5
        elif self.controller[j][i] == SW:
            if ( sensors[SW] ):
                return 1.5
            elif ( sensors[NW] ):
                return 0.5            
            else:
                return 0.5
        elif self.controller[j][i] == SE:
            if ( sensors[SE] ):
                return 1.5
            elif ( sensors[NE] ):
                return 0.5            
            else:
                return 0.5
        elif self.controller[j][i] == S:
            if ( sensors[SW] or sensors[SE] ):
                return 0.5
            elif ( sensors[NE] or sensors[NW] ):
                return 1.5
            else:
                return 0.5
        else:
            return 0.5
    
    def update( self, i, j, timestep, sensors):
        
        return( self.square( i, j, sensors ) )

