
from lib2to3.pytree import Node
from sre_parse import State
import numpy as np
from itertools import count
import pprint




class Property():
    def __init__(self, *arg):

        
        self.pre = np.array([

                            [3, "g"],
                            [5, "O"],
                            [4, "F"],
                            [7, "E"],
                            [2, "D"],
                            [5, "C"],
                            [3, "B"],
                            [4, "A"],
                            [0, "s"]
                            # [3*2, "g"],
                            # [5*2, "O"],
                            # [4*2, "F"],
                            # [7*2, "E"],
                            # [2*2, "D"],
                            # [5*2, "C"],
                            # [3*2, "B"],
                            # [4*2, "A"],
                            # [0, "s"]

                            ])

    
    def reference(self):
        
        
        Node = self.pre[:, 1]
        Arc = self.pre[:, 0]
        print(Node)
        print(Arc)
        Node = Node.tolist()
        Arc = Arc.tolist()
        num = [float(i) for i in Arc]
        print(type(num))
        Arc_sum = sum(num)
        print(Arc_sum)

        PERMISSION = [
                # [0],
                # [2],
                # [5],
                # [7],
                # [10]
                # [Arc_sum-int(Arc[4])-int(Arc[3])-int(Arc[2])-int(Arc[1])-int(Arc[0])],
                # [Arc_sum-int(Arc[4])-int(Arc[3])-int(Arc[2])-int(Arc[1])],
                # [Arc_sum-int(Arc[4])-int(Arc[3])-int(Arc[2])],
                # [Arc_sum-int(Arc[4])-int(Arc[3])],
                # [Arc_sum-int(Arc[4])],
                # [Arc_sum]
                # [Arc_sum-float(Arc[4])-float(Arc[3])-float(Arc[2])-float(Arc[1])-float(Arc[0])],
                # [Arc_sum-float(Arc[4])-float(Arc[3])-float(Arc[2])-float(Arc[1])],
                # [Arc_sum-float(Arc[4])-float(Arc[3])-float(Arc[2])],
                # [Arc_sum-float(Arc[4])-float(Arc[3])],
                # [Arc_sum-float(Arc[4])],
                # [Arc_sum]

                # [0],
                # [5],
                # [5],
                # [5],
                # [5],
                # [8]


                # [0],
                # [10],
                # [10],
                # [12],
                # [20],
                # [30]
                # [Arc_sum-float(Arc[5])-float(Arc[4])-float(Arc[3])-float(Arc[2])-float(Arc[1])-float(Arc[0])],
                # [Arc_sum-float(Arc[5])-float(Arc[4])-float(Arc[3])-float(Arc[2])-float(Arc[1])],
                # [Arc_sum-float(Arc[5])-float(Arc[4])-float(Arc[3])-float(Arc[2])],
                # [Arc_sum-float(Arc[5])-float(Arc[4])-float(Arc[3])],
                # [Arc_sum-float(Arc[5])-float(Arc[4])],
                # [Arc_sum-float(Arc[5])],
                # [Arc_sum]

                # [0],
                # [7],
                # [8],
                # [11],
                # [4]

                # [0],
                # [5],
                # [8],
                # [10],
                # [4]

                # Add 1027
                # [0],
                # [3],
                # [8],
                # [12],
                # [19],
                # [21],
                # [26],
                # [29],
                # [33]
                [0],
                [5],
                [8],
                [12],
                [19],
                [21],
                [26],
                [29],
                [33]
        ]

        return self.pre, Node, Arc, Arc_sum, PERMISSION