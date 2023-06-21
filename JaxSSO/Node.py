"""
References
---------
1. PyNite: An open-source FEA solver for python; https://github.com/JWock82/PyNite
2. Bathe, K. J. (2006). Finite element procedures. Klaus-Jurgen Bathe.
"""      


class Node():
    """
    Create a JaxSSO node.

    Input
    -----
    nodeTag: int 
        the tag/index of this node
    X, Y, Z: float
        the coordinates of thie node

    """
    
    def __init__(self, nodeTag, X, Y, Z):
        self.nodeTag = nodeTag      # Index of this node
        self.X = X            # Global X coordinate
        self.Y = Y            # Global Y coordinate
        self.Z = Z            # Global Z coordinate
    

