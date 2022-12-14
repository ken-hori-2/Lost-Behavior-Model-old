from pprint import pprint
import numpy as np
import pprint
from reference_match_rate_Robosin import Property


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
        self.Stressfull = 2.0
        self.COUNT = 0
        self.done = False
        self.TRIGAR = False
        ########## parameter ##########
        self.STATE_HISTORY = []
        self.bp_end = False
        self.test_s = 0

    def Explore(self, STATE_HISTORY, state, TRIGAR, total_stress, grid, CrossRoad, x, TOTAL_STRESS_LIST): # , PERMISSION):

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

        self.move_step = 0
        self.old_to_advance = self.NODELIST[self.state.row][self.state.column]

        
        while not self.done:
            print("\n========== 🌟 {}steps ==========".format(self.COUNT+1))
            print(f"🤖 State:{self.state}")
            print("stress : {}".format(self.stress))

            # if not self.crossroad:
            self.map_unexp_area = self.env.map_unexp_area(self.state)
            if self.map_unexp_area:
                print("un explore area ! 🤖 ❓❓")
                
                if self.total_stress + self.stress >= 0:
                    # if self.test_s + self.stress >= 0:
                    # self.total_stress += self.stress
                    try:
                        self.test_s += round(self.stress/float(Arc[index-1]), 3) # 2)
                        self.total_stress += round(self.stress/float(Arc[index-1]), 3)

                        "----- min cost cal -----"
                        if not self.NODELIST[self.state.row][self.state.column] in pre:
                            self.move_step += 1
                        "----- min cost cal -----"
                    except:
                        self.test_s += 0
                        self.total_stress += 0

                        "----- min cost cal -----"
                        self.move_step += 0
                        "----- min cost cal -----"
                    print("Arc to the next node : {}".format(Arc[index-1]))
                    "---- Add 1108 ----"

                if self.NODELIST[self.state.row][self.state.column] in pre:

                    print("🪧 NODE : ⭕️")

                    "expで増加した分だけ減少"
                    if self.total_stress - self.test_s >= 0:
                        self.total_stress -= self.test_s
                    else:
                        self.total_stress = 0
                    self.test_s = 0
                    "expで増加した分だけ減少"


                    if self.NODELIST[self.state.row][self.state.column] == "g":
                        print("🤖 GOALに到達しました。")
                        GOAL = True
                        self.STATE_HISTORY.append(self.state)
                        self.TOTAL_STRESS_LIST.append(self.total_stress)
                        break
                    # self.TRIGAR = False # ここでFalseにすることでadvance_Algorithmで撮った場所のノードも追加してしまう
                    print("\n============================\n🤖 🔛　アルゴリズム切り替え\n============================")
                    break # Advanceに移行する？
                else:
                    if self.grid[self.state.row][self.state.column] == 5:
                        print("\n\n\n交差点! 🚥　🚙　✖️")
                        if self.state not in self.CrossRoad:
                            print("\n\n\n未探索の交差点! 🚥　🚙　✖️")
                            self.CrossRoad.append(self.state)

                    print("CrossRoad : {}\n\n\n".format(self.CrossRoad))
                    print("🪧 NODE : ❌")
                    print("no match!")

            "ここはあってもあまり変わらない? 一応知っているところにたどり着いたらストレスを減少させておくが、たぶんmark=1になっている場所には進まないから意味ない"
            if self.NODELIST[self.state.row][self.state.column] in pre:
                index = Node.index(self.NODELIST[self.state.row][self.state.column])
                print("<{}> match !".format(self.NODELIST[self.state.row][self.state.column]))
                print("事前のArc : {}".format(Arc[index]))
                # self.total_stress = 0
                "expで増加した分だけ減少"
                if self.total_stress - self.test_s >= 0:
                    self.total_stress -= self.test_s
                else:
                    self.total_stress = 0
                self.test_s = 0
                "expで増加した分だけ減少"

            print("PERMISSION : {}".format(PERMISSION[index][0]))
            # if self.total_stress >= PERMISSION[index][0]                +x: # or self.lost:　　　　# 追加
            if self.total_stress >= self.Stressfull: # 2.0:
                self.TRIGAR = True
                print("=================")
                print("FULL ! MAX! 🔙⛔️")
                print("=================")
                self.state = self.NODE_POSITION
                print(f"🤖 State:{self.state}")
                # self.total_stress = 0
                self.STATE_HISTORY.append(self.state)
                self.TOTAL_STRESS_LIST.append(self.total_stress)
                print(f"Total Stress:{self.total_stress}")

                "expで増加した分だけ減少"
                if self.total_stress - self.test_s >= 0:
                    self.total_stress -= self.test_s
                else:
                    self.total_stress = 0
                self.test_s = 0
                "expで増加した分だけ減少"

                "Advance ver."
                # standard = []
                # standard.append(self.test_s)
                # print("standard【基準距離】 : {}".format(standard[0]))
                # if standard[0] != 0:
                #     "-- これがいずれのΔSnodeの式 今はArc に対するΔSのみ --"
                #     arc_s = round(abs(1.0-standard[0]), 2)
                #     if arc_s > 2:
                #         arc_s = 1.0
                # else:
                #     arc_s = 0.5 # 0.0
                # print("arc stress【基準ストレス】 : {}".format(arc_s))  #このままだとArcが大きくなるとストレス値も大きくなってしまい、ストレス値の重みが変わってしまうので、基準[1]にする 
            
                "Advance とは違い、こっちにはbreakはない"


                "----- min cost cal -----"
                self.move_step = 0
                "----- min cost cal -----"

            print(f"Total Stress:{self.total_stress}")
            print("trigar : {}".format(self.TRIGAR))
            self.STATE_HISTORY.append(self.state)
            self.TOTAL_STRESS_LIST.append(self.total_stress)

            self.action, self.bp_end, self.All_explore, self.TRIGAR, self.Reverse, self.lost = self.agent.policy_exp(self.state, self.TRIGAR)
            if self.lost:
                self.TRIGAR = True
                print("=================")
                print("LOST! 🔙⛔️")
                print("=================")
                self.state = self.NODE_POSITION
                self.STATE_HISTORY.append(self.state)
                self.TOTAL_STRESS_LIST.append(self.total_stress)
                print(f"🤖 State:{self.state}")
                # self.total_stress = 0 # TEST 1108
                
                "expで増加した分だけ減少"
                if self.total_stress - self.test_s >= 0:
                    self.total_stress -= self.test_s
                else:
                    self.total_stress = 0
                self.test_s = 0
                "expで増加した分だけ減少"

                "Add1108"
                # break

                "----- min cost cal -----"
                self.move_step = 0
                "----- min cost cal -----"
                
            print("All explore : {}".format(self.All_explore))
            if self.All_explore:
                self.env.mark_all(state)
                print("終了します")
                self.All_explore = False
                # self.total_stress = 0

                ############コメントアウト##############
                # self.TRIGAR = True
                ############コメントアウト##############

                "----- min cost cal -----"
                self.move_step = 0
                # self.old_to_advance = self.NODELIST[self.state.row][self.state.column]
                "----- min cost cal -----"
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

        return self.total_stress, self.STATE_HISTORY, self.state, self.TRIGAR, self.CrossRoad, GOAL, self.TOTAL_STRESS_LIST, self.move_step, self.old_to_advance