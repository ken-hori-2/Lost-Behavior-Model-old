from pprint import pprint
import numpy as np
import pprint
from reference_match_rate import Property
import random


class Algorithm_exp():

    def __init__(self, *arg):
        
        self.state = arg[0] # state
        self.env = arg[1] # env
        self.agent = arg[2] # agent
        self.NODELIST = arg[3] # NODELIST
        self.Observation = arg[4]
        self.refer = Property() # arg[5]
        ########## parameter ##########
        self.total_stress = 0
        self.stress = 0
        self.Stressfull = 8 # 2.0
        self.COUNT = 0
        self.done = False
        self.TRIGAR = False
        ########## parameter ##########
        self.STATE_HISTORY = []
        self.bp_end = False
            
    def Explore(self, STATE_HISTORY, state, TRIGAR, total_stress, grid, CrossRoad, x, TOTAL_STRESS_LIST, Node_s, Node_A, Node_B, Node_C, Node_D, Node_g, Cost_S, Cost_O, Cost_A, Cost_B, Cost_C, Cost_D, WEIGHT_CROSS_S, WEIGHT_CROSS_O, WEIGHT_CROSS_A, WEIGHT_CROSS_B, WEIGHT_CROSS_C, WEIGHT_CROSS_D): # , PERMISSION):

        self.STATE_HISTORY = STATE_HISTORY
        self.state = state
        self.TRIGAR = False # TRIGAR
        pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()
        pprint.pprint(PERMISSION)
        self.NODE_POSITION = state
        # self.map_unexp_area = map_unexp_area
        self.lost = False
        self.grid = grid
        self.CrossRoad = CrossRoad
        GOAL = False
        "-- 追加部分 --"
        # self.total_stress = 0
        self.total_stress = total_stress
        "-- 追加部分 --"


        self.stress = 0
        # 初期
        index = Node.index("s")
        self.TOTAL_STRESS_LIST = TOTAL_STRESS_LIST
        self.Node_s = Node_s
        self.Node_A = Node_A
        self.Node_B = Node_B
        self.Node_C = Node_C
        self.Node_D = Node_D
        self.Node_g = Node_g

        self.Cost_S = Cost_S
        self.Cost_O = Cost_O
        self.Cost_A = Cost_A
        self.Cost_B = Cost_B
        self.Cost_C = Cost_C
        self.Cost_D = Cost_D

        self.WEIGHT_CROSS_S = WEIGHT_CROSS_S
        self.WEIGHT_CROSS_O = WEIGHT_CROSS_O
        self.WEIGHT_CROSS_A = WEIGHT_CROSS_A
        self.WEIGHT_CROSS_B = WEIGHT_CROSS_B
        self.WEIGHT_CROSS_C = WEIGHT_CROSS_C
        self.WEIGHT_CROSS_D = WEIGHT_CROSS_D

        while not self.done:
            print("\n========== 🌟 {}steps ==========".format(self.COUNT+1))
            print(f"🤖 State:{self.state}")
            print("stress : {}".format(self.stress))

            # if not self.crossroad:
            self.map_unexp_area = self.env.map_unexp_area(self.state)
            if self.map_unexp_area:
                print("un explore area ! 🤖 ❓❓")
                if self.NODELIST[self.state.row][self.state.column] in pre:

                    print("🪧 NODE : ⭕️")
                    if self.total_stress + self.stress >= 0:
                        self.total_stress += self.stress
                    if self.NODELIST[self.state.row][self.state.column] == "g":
                        print("🤖 GOALに到達しました。")
                        GOAL = True
                        self.STATE_HISTORY.append(self.state)
                        self.TOTAL_STRESS_LIST.append(self.total_stress)
                        break
                    print("\n============================\n🤖 🔛　アルゴリズム切り替え\n============================")
                    break

            
            if self.NODELIST[self.state.row][self.state.column] in pre:
                index = Node.index(self.NODELIST[self.state.row][self.state.column])
                print("<{}> match !".format(self.NODELIST[self.state.row][self.state.column]))
                print("事前のArc : {}".format(Arc[index]))
                # self.total_stress = 0
                "-- 追加部分 --"
                # delta_s = self.Observation[self.state.row][self.state.column]
                # self.total_stress -= (1-delta_s)
                "-- 追加部分 --"


            else:
                if self.total_stress + self.stress >= 0:
                    self.total_stress += self.stress
                if self.grid[self.state.row][self.state.column] == 5:
                    print("\n\n\n交差点! 🚥　🚙　✖️")
                    if self.state not in self.CrossRoad:
                        print("\n\n\n未探索の交差点! 🚥　🚙　✖️")
                        self.CrossRoad.append(self.state)

                    print("CrossRoad : {}\n\n\n".format(self.CrossRoad))

            print("PERMISSION : {}".format(PERMISSION[index][0]))
            # if self.total_stress >= self.Stressfull:
            if self.total_stress >= PERMISSION[index][0]                +x: # or self.lost:　　　　# 追加
                self.TRIGAR = True
                print("=================")
                print("FULL ! MAX! 🔙⛔️")
                print("=================")
                self.state = self.NODE_POSITION
                print(f"🤖 State:{self.state}")
                # self.total_stress = 0
                "-- 追加部分 --"
                # delta_s = self.Observation[self.state.row][self.state.column]
                # self.total_stress -= (1-delta_s)
                "-- 追加部分 --"


            print(f"Total Stress:{self.total_stress}")
            print("trigar : {}".format(self.TRIGAR))

            "--------------------------------------------------"
            self.STATE_HISTORY.append(self.state)
            self.TOTAL_STRESS_LIST.append(self.total_stress)
            # self.TOTAL_STRESS_LIST.append(abs(1.0-self.total_stress))

            self.STATE_HISTORY.append(self.state)
            self.total_stress = round(random.uniform(2.0,2.5), 3) # 2 # 10
            self.TOTAL_STRESS_LIST.append(self.total_stress) # 分岐先を探索した前提
            # self.TOTAL_STRESS_LIST.append(abs(1.0-self.total_stress))
            
            self.Node_s.append(0)
            self.Node_A.append(0)
            self.Node_B.append(0)
            self.Node_C.append(0)
            self.Node_D.append(0)
            self.Node_g.append(0)

            # Add 1029
            self.Node_s.append(0)
            self.Node_A.append(0)
            self.Node_B.append(0)
            self.Node_C.append(0)
            self.Node_D.append(0)
            self.Node_g.append(0)

            "--test--"
            # self.Cost_S.append(0)
            # self.Cost_A.append(0)
            # self.Cost_B.append(0)
            # self.Cost_C.append(0)
            # self.Cost_D.append(0)
            # self.Cost_O.append(0)

            # self.WEIGHT_CROSS_S.append(0)
            # self.WEIGHT_CROSS_O.append(0)
            # self.WEIGHT_CROSS_A.append(0)
            # self.WEIGHT_CROSS_B.append(0)
            # self.WEIGHT_CROSS_C.append(0)
            # self.WEIGHT_CROSS_D.append(0)
            "--------------------------------------------------"



            self.action, self.bp_end, self.All_explore, self.TRIGAR, self.Reverse, self.lost = self.agent.policy_exp(self.state, self.TRIGAR)
            if self.lost:
                self.TRIGAR = True
                print("=================")
                print("LOST! 🔙⛔️")
                print("=================")
                self.state = self.NODE_POSITION
                print(f"🤖 State:{self.state}")
                # self.total_stress = 0
                "-- 追加部分 --"
                # delta_s = self.Observation[self.state.row][self.state.column]
                # self.total_stress -= (1-delta_s)
                "-- 追加部分 --"


            print("All explore : {}".format(self.All_explore))
            if self.All_explore:
                self.env.mark_all(state)
                print("終了します")
                self.All_explore = False
                ############コメントアウト##############
                # self.TRIGAR = True
                ############コメントアウト##############
                break
            # self.next_state, self.stress, self.done = self.env._move(self.state, self.action, self.TRIGAR, self.All_explore, self.Reverse)
            if not self.lost:
                # self.next_state, self.stress, self.done = self.env._move(self.state, self.action, self.TRIGAR)
                self.next_state, self.stress, self.done = self.env.step(self.state, self.action, self.TRIGAR)
                self.prev_state = self.state # 1つ前のステップを保存 -> 後でストレスの減少に使う
                self.state = self.next_state
            else:
                self.lost = False
            if self.COUNT > 150:
                break
            self.COUNT += 1

        if self.done:
            print("GOAL")

        return self.total_stress, self.STATE_HISTORY, self.state, self.TRIGAR, self.CrossRoad, GOAL, self.TOTAL_STRESS_LIST, self.Node_s, self.Node_A, self.Node_B, self.Node_C, self.Node_D, self.Node_g, self.Cost_S, self.Cost_O, self.Cost_A, self.Cost_B, self.Cost_C, self.Cost_D, self.WEIGHT_CROSS_S, self.WEIGHT_CROSS_O, self.WEIGHT_CROSS_A, self.WEIGHT_CROSS_B, self.WEIGHT_CROSS_C, self.WEIGHT_CROSS_D