import numpy as np
from constants import *

class controller:
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def update( self, i, j, timestep, sensors ):
        if ( i == 0 ):
            return( 1.0 )
        elif ( i == 1 ):
            if( j == 0 ):                
                if( sensors[NW] < TARGET_WEIGHT ):
                    return( 0.5 )
                else:
                    return( 1.5 )
            if( j == 1 ):
                if( sensors[SW] < TARGET_WEIGHT ):
                    return( 0.5 )
                else:
                    return( 1.5 )
        elif( i > 1 ):
            if( j == 0 ):                
                if( sensors[NW] > 0 ):
                    return( 1.5 )
                else:
                    return( 0.5 )
            if( j == 1 ):
                if( sensors[SW] > 0 ):
                    return( 1.5 )
                else:
                    return( 0.5 )

     
 

