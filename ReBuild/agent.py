from cgi import test
from tkinter.messagebox import NO
import numpy as np
from sklearn import preprocessing
import random
from reference_match_rate_Robosin import Property
import math
from Lost_Action_actions import Agent_actions
import pandas as pd


class Agent():

    # def __init__(self, env, GOAL_STATE, NODELIST, map, grid):
    def __init__(self, env, marking_param, *arg):
        self.env = env
        self.actions = env.actions
        self.GOAL_REACH_EXP_VALUE = 50 # max_theta # 50
        self.lost = False
        self.test = False
        self.grid = arg[0]
        self.map = arg[1]
        self.NODELIST = arg[2]
        # self.goal = arg[3]
        self.refer = Property() # arg[5]
        # self.actions = self.env.actions
        # self.goal = GOAL_STATE
        # self.NODELIST = NODELIST
        # self.map = map
        # self.grid = grid
        # print("GOAL STATE : {}".format(self.goal))
        self.marking_param = marking_param

        "======================================================="
        self.decision_action = Agent_actions(self.env)
        "======================================================="
        self.Node_l = ["s", "A", "B", "C", "D", "E", "F", "O", "g", "x"]

    def policy_advance(self, state, TRIGAR, action):
        
        self.TRIGAR_advance = TRIGAR
        self.prev_action = action
        print("Prev Action : {}".format(action))

        action = self.model_advance(state)
        self.Advance_action = action

        print("Action : {}".format(action))
        print("Advance action : {}".format(self.Advance_action))
        print("๐ ๐ ๐ ๐ ๐")

        if action == None:
            print("ERROR ๐ค")
            ############ใณใกใณใใขใฆใ##############
            # self.TRIGAR_advance = True
            ############ใณใกใณใใขใฆใ##############
            # return self.actions[1], self.Reverse, self.TRIGAR_advance # ใใฎaction[1]ใใจใฉใผใฎๅๅ 
            return self.prev_action, self.Reverse, self.TRIGAR_advance # ใใฎprev action ใไปฎ
            
        return action, self.Reverse, self.TRIGAR_advance

    def policy_bp(self, state, TRIGAR, TRIGAR_REVERSE, COUNT):
        self.TRIGAR_bp = TRIGAR
        self.TRIGAR_REVERSE_bp = TRIGAR_REVERSE
        self.All = False
        self.Reverse = False
        # self.lost = False
        self.COUNT = COUNT

        try:
            # self.lost = False
            action, self.Reverse = self.model_bp(state)
            print("Action : {}".format(action))
        except:
        # except Exception as e:
        #     print('=== ใจใฉใผๅๅฎน ===')
        #     print('type:' + str(type(e)))
        #     print('args:' + str(e.args))
        #     print('message:' + e.message)
        #     print('e่ช่บซ:' + str(e))
            print("agent / policy_bp ERROR")

            "ๅใใฆใใชใๆใซ่ฟทใฃใใจใใๅ ดๅ"
            # if NOT_MOVE:
            #     self.All = True
            
            # ใใใฎใใใใงๆฒผใงใๅฐใๅใใฆใใ
            # return self.actions[1], self.Reverse    , self.lost
            return random.choice(self.actions), self.Reverse    , self.lost
        print("๐ ๐ ๐ ๐ ๐")
        # return action, self.All, self.Reverse
        return action, self.Reverse , self.lost

    def policy_exp(self, state, TRIGAR):
        self.trigar = TRIGAR
        attribute = self.NODELIST[state.row][state.column]
        next_direction = random.choice(self.actions)
        self.All = False
        bp = False
        self.lost = False
        self.Reverse = False
        
        try:
            y_n, action, bp = self.model_exp(state)
            print("y/n:{}".format(y_n))
            print("Action : {}".format(action))
        except:
            print("ใใฎใใผใใใๆข็ดขใงใใ่จฑๅฎน็ฏๅฒใฏๆข็ดขๆธใฟ\nๆปใๅ ดๆๆฑบๅฎใฎใขใซใดใชใบใ ใธ")
            print("TRIGAR : {}".format(self.trigar))
            # self.All = True
            return self.actions[1], bp, self.All, self.trigar, self.Reverse, self.lost
        return action, bp, self.All, self.trigar, self.Reverse, self.lost

    def model_exp(self, state):

        next_diretion = [(self.actions[0]), (self.actions[1]), (self.actions[2]), (self.actions[3])]

        y_n = False
        bp = False
        pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()

        # ไปใฏใใใซๅฅใฃใฆbp_algorithmใซ้ท็งปใใฆใใ
        if self.NODELIST[state.row][state.column] in pre:
                print("========\nๆข็ดข็ตไบ\n========")
                self.trigar = False
                bp = True
        elif self.NODELIST[state.row][state.column] == "x":
            print("========\nไบคๅทฎ็น\n========")
            self.trigar = False

        print("========\nๆข็ดข้ๅง\n========")
        # if not self.trigar:
        exp_action = [] # Add 1108
        for dir in next_diretion:

            print("dir:{}".format(dir))
            y_n, action = self.env.expected_move(state, dir, self.trigar, self.All, self.marking_param)
            
            if y_n:
                y_n = False
                exp_action.append(action)
                print("================================================== exp action : {}".format(exp_action))
            # if y_n:
            #     return y_n, action, bp
            # print("y/n:{}".format(y_n))
        if exp_action:

            "======================================================="
            "----- Add 1110 -----"
            if self.NODELIST[state.row][state.column] in pre: # "x":
                print("========\nไบคๅทฎ็น\n========")
                ##############
                Average_Value = self.decision_action.value(exp_action)
                ##############
                print("\n===================\n๐คโก๏ธ Average_Value:{}".format(Average_Value))
                print(" == ๅ่กๅๅพใซในใใฌในใๆธใใใ็ขบ็:{}".format(Average_Value))
                print(" == ใคใพใใๆฐใใๆๅ ฑใๅพใใใ็ขบ็:{} -----> ใใใไธ็ช้่ฆใปใปใปๆชๆข็ดขใใคใใฎๆฐๅคใๅคงใใๆนๅใฎ่กๅใ้ธๆ\n===================\n".format(Average_Value))
                ##############
                action_value = self.decision_action.policy(Average_Value)
                ##############
                if action_value == self.env.actions[2]: #  LEFT:
                    NEXT = "LEFT  โฌ๏ธ"
                    print("    At :-> {}".format(NEXT))
                if action_value == self.env.actions[3]: # RIGHT:
                    NEXT = "RIGHT โก๏ธ"
                    print("    At :-> {}".format(NEXT))  
                if action_value == self.env.actions[0]: #  UP:
                    NEXT = "UP    โฌ๏ธ"
                    print("    At :-> {}".format(NEXT))
                if action_value == self.env.actions[1]: # DOWN:
                    NEXT = "DOWN  โฌ๏ธ"
                    print("    At :-> {}".format(NEXT))

                print("้ๅปใฎใจใใฝใผใใใใ็พๆ็นใงใฏใ๐คโ ๏ธ At == {}ใ้ธๆใใ".format(action_value))
                ##############
                Episode_0 = self.decision_action.save_episode(action_value)
                ##############
                # print("\n===================\n๐คโก๏ธ Average_Value:{}".format(Average_Value))
                # print(" == ๅ่กๅๅพใซในใใฌในใๆธใใใ็ขบ็:{}".format(Average_Value))
                # print(" == ใคใพใใๆฐใใๆๅ ฑใๅพใใใ็ขบ็:{} -----> ใใใไธ็ช้่ฆใปใปใปๆชๆข็ดขใใคใใฎๆฐๅคใๅคงใใๆนๅใฎ่กๅใ้ธๆ\n===================\n".format(Average_Value))
            else:
                action_value = exp_action[0]
            "----- Add 1110 -----"
            "======================================================="

            for x in exp_action:
                print("1015 exp action : {}".format(x))
            y_n = True
            # return y_n, exp_action[0], bp
            "======================================================="
            "----- Add 1110 -----"
            return y_n, action_value, bp
            "======================================================="
        print("y/n:{}".format(y_n))

        if not bp:
            print("==========\nใใไปฅไธ้ฒใใชใ็ถๆ\n or ๆฌกใฎใในใฏๆข็ดขๆธใฟ\n==========") # ใฉใฎ้ธๆ่ขใ y_n = False
            self.lost = True
            # self.trigar = True
        else:
            self.All = True # False
                # for dir in next_diretion:
                #     print("\ndir:{}".format(dir))
                #     y_n, action = self.env.expected_move_return(state, dir, self.trigar, self.All)

                #     if y_n:
                #         return y_n, action, bp
                #     print("y/n:{}".format(y_n))
        # else:
        #     for dir in next_diretion:
        #         print("\ndir:{}".format(dir))
        #         y_n, action = self.env.expected_move_return(state, dir, self.trigar, self.All)

        #         if y_n:
        #             return y_n, action, bp
        #         print("y/n:{}".format(y_n))

        print("==========\n่ฟทใฃใ็ถๆ\n==========") # ใฉใฎ้ธๆ่ขใ y_n = False
        print("= ็พๅจๅฐใใใดใผใซใซ่ฟใใ้ธๆ่ขใฏใชใ\n")
        # self.lost = True

    def model_advance(self, state):

        next_diretion = [(self.actions[0]), (self.actions[1]), (self.actions[2]), (self.actions[3])]
        # next_diretion = [(self.actions[1]), (self.actions[0]), (self.actions[2]), (self.actions[3])]
        # next_diretion = []

        pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()
        if self.NODELIST[state.row][state.column] in pre:
            print("ใฉใณใใ ใซๆฑบๅฎ")
            next_diretion = self.advance_direction_decision(next_diretion)
        else:
            next_diretion = [(self.actions[0]), (self.actions[1]), (self.actions[2]), (self.actions[3])]
        # ่ฉฆใใซใณใกใณใใขใฆใ0923
        # advanceใฎ่กๅใฎๅชๅๅบฆใใใใใใ่จญๅฎ

        if self.NODELIST[state.row][state.column] == "x":
            print("ใฉใณใใ ใซๆฑบๅฎ")
            next_diretion = self.advance_direction_decision(next_diretion)
        print("next dir : {}".format(next_diretion))

        y_n = False
        # bp = False
        self.All = False
        self.Reverse = False

        if self.NODELIST[state.row][state.column] == "x":
            print("========\nไบคๅทฎ็น\n========")
            self.TRIGAR_advance = False

        print("========\nAdvance้ๅง\n========")
        if not self.TRIGAR_advance:
            for dir in next_diretion:

                print("dir:{}".format(dir))
                y_n, action = self.env.expected_move(state, dir, self.TRIGAR_advance, self.All, self.marking_param)
                # self.prev_action = action
                # print("prev action : {}".format(self.prev_action))
                
                if y_n:
                    self.prev_action = action
                    # print("prev action : {}".format(self.prev_action))
                    return action
                print("y/n:{}".format(y_n))
        print("==========\n่ฟทใฃใใ่จฑๅฎนใ่ถใใใ็ถๆ\n==========") # ใฉใฎ้ธๆ่ขใ y_n = False
        print("= ใใไปฅไธๅใซ็พๅจๅฐใใใดใผใซใซ่ฟใใ้ธๆ่ขใฏใชใ\n= ไธๆฆไฝๅถใๆดใใ\n= ๆปใ")
        print("\n ใจใใใใใฏในใใฌในใๆบใพใๅใๅใซใใไปฅไธ้ฒใใชใใชใฃใฆใจใฉใผใๅบใ")
        self.TRIGAR_advance = True
        # # self.trigar = True
        # print("prev action2 : {}".format(self.prev_action))
        # lost = True
        # return self.prev_action

    def model_bp(self, state):

        # next_diretion = [(self.actions[0]), (self.actions[1]), (self.actions[2]), (self.actions[3])]
        # next_diretion = [(self.actions[1]), (self.actions[0]), (self.actions[2]), (self.actions[3])]

        # if self.NODELIST[state.row][state.column] == "x":
        #     print("========\nไบคๅทฎ็น\n========")
        #     self.TRIGAR_bp = False
        #     self.TRIGAR_REVERSE_bp = False
        pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()

        # if self.COUNT > 100: # 40:
        #     # if self.NODELIST[self.state.row][self.state.column] in pre:
        #     #     pass
        #     print("ๆฒผใซใใใฃใๆใซใจใใใใ1ใฎใใผใญใณใฐ้ใใซๆปใๆฉ่ฝ")
        #     next_diretion = self.next_direction_decision("trigar")
        #     for dir in next_diretion:
        #         print("\ndir:{}".format(dir))
        #         y_n, action = self.env.expected_move_return(state, dir, self.TRIGAR_bp, self.All)

        #         if y_n:
        #             self.lost = False
        #             return action, self.Reverse

        #     print("ใใผใญใณใฐใ1ใฎๆนๅใฏใฉใใซใใชใ -> ใใผใญใณใฐ2ใฎๆนๅใ็ฎๆใ")
        #     for dir in next_diretion:
        #         print("\ndir:{}".format(dir))
        #         y_n, action = self.env.expected_move_return_reverse(state, dir, self.TRIGAR_REVERSE_bp, self.Reverse)

        #         if y_n:
        #             self.lost = False
        #             return action, self.Reverse

        print("========\nBACK ้ๅง\n========")
        print("TRIGAR : {}".format(self.TRIGAR_bp))
        print("REVERSE : {}".format(self.TRIGAR_REVERSE_bp))
        
        if self.TRIGAR_REVERSE_bp:
            self.Reverse = True
            # next_diretion = self.next_direction_trigar() # [(self.actions[0]), (self.actions[1]), (self.actions[2]), (self.actions[3])]
            next_diretion = self.next_direction_decision("reverse")
            for dir in next_diretion:
                print("\ndir:{}".format(dir))
                y_n, action = self.env.expected_move_return_reverse(state, dir, self.TRIGAR_REVERSE_bp, self.Reverse)

                if y_n:
                    self.lost = False
                    return action, self.Reverse
                print("y/n:{}".format(y_n))
            print("TRIGAR REVERSE โก๏ธ๐")

        if self.TRIGAR_bp:
            # next_diretion = self.next_direction_trigar_reverse() # [(self.actions[1]), (self.actions[0]), (self.actions[2]), (self.actions[3])]
            next_diretion = self.next_direction_decision("trigar")
            print("TEST!!!!!")

            # self.TRIGAR_REVERSE_bp = False
            for dir in next_diretion:
                print("\ndir:{}".format(dir))
                y_n, action = self.env.expected_move_return(state, dir, self.TRIGAR_bp, self.All)

                if y_n:
                    self.lost = False
                    return action, self.Reverse
                print("y/n:{}".format(y_n))

            # if not bp:
            if self.lost:
                print("==========\nใใไปฅไธๆปใใชใ็ถๆ\n or ๆฌกใฎใในใฏไปฅๅๆปใฃใๅ ดๆ\n==========") # ใฉใฎ้ธๆ่ขใ y_n = False
                # self.lost = False
                # self.trigar = True
                for dir in next_diretion:
                    print("\ndir:{}".format(dir))
                    y_n, action = self.env.expected_not_move(state, dir, self.trigar, self.All)

                    if y_n:
                        return action, self.Reverse # , self.lost
                    print("y/n:{}".format(y_n))

        print("==========\nๆปใ็ตใใฃใ็ถๆ\n==========") # ใฉใฎ้ธๆ่ขใ y_n = False
        print("= ็พๅจๅฐใใๆฌกใซใดใผใซใซ่ฟใใ้ธๆ่ขใ้ธใถใๆชๆข็ดขๆนๅใ\n")
        self.lost = True

    # def back_position(self, BPLIST, w, Arc):
    def back_position(self, BPLIST, w, Arc, Cost): # change
        
        "----------------------------------------------------------------------"
        # ในใใฌในใฎๅฐใใใใผใใซๆปใver.
        "== stressใฎๅฐใใใงๆปใใใผใใๆฑบใใๅ ดๅ =="
        # Move_Cost = [round(Arc[x],2) for x in range(len(Arc))] # Arc_INVERSE ใงใฏใชใ Arc
        Move_Cost = [round(Cost[x],2) for x in range(len(Cost))] # change
        "----------------------------------------------------------------------"  
        "----------------------------------------------------------------------"
        # ๆญฃ่ฆๅใซใใใจ0, 1ใๅบใฆใใพใใฎใงใstressรcost ใง0ใซใชใใใใใใใใซๆปใใใจใๅคใใชใฃใฆใใพใ 1026
        "ๆญฃ่ฆๅใฎ็บใฎๅฆ็"  
        # w = np.round(preprocessing.minmax_scale(w), 3)
        # Arc = np.round(preprocessing.minmax_scale(Arc), 3)
        # Move_Cost = np.round(preprocessing.minmax_scale(Move_Cost), 3)
        "----------------------------------------------------------------------"
        # print("๐ๆญฃ่ฆๅ w : {}, Arc : {}".format(w, Arc))
        # print("๐ๆญฃ่ฆๅ Weight : {}, Move Cost : {}".format(w, Move_Cost))
        print("๐ ๆญฃ่ฆๅ WEIGHT : {}, Move_Cost : {}".format(w, Move_Cost))
        print(type(w), type(Move_Cost))
        "-> ใฉใฃใกใlist"

        # Arc = [0, 0]ใฎๆ,Arc = [1, 1]ใซๅคๆด
        if all(elem  == 0 for elem in Move_Cost):
            Move_Cost = [1 for elem in Move_Cost]
            print("   Arc = [0, 0]ใฎๆ, Move_Cost : {}".format(Move_Cost))
        if all(elem  == 0 for elem in w):
            w = [1 for elem in w]
            print("   WEIGHT = [0, 0]ใฎๆ, WEIGHT : {}".format(w))

        WEIGHT_CROSS = [round(x*y, 3) for x,y in zip(w,Move_Cost)]
        "->ๆน่ฏใใๅฟ่ฆใใ"
        "OBSใฎใฟๅ้คใใใฆใใ"

        
        print("โก๏ธ WEIGHT CROSS:{}".format(WEIGHT_CROSS))

        try:
            if all(elem  == 0 for elem in WEIGHT_CROSS):
                print("WEIGHT CROSSใฏๅจ้จ0ใงใใ")
                
                # Arc = Arc.tolist()
                print("Arc type : {}".format(type(Arc)))
                near_index = Arc.index(min(Arc))
                print("Arc:{}, index:{}".format(Arc, near_index))
                WEIGHT_CROSS[near_index] = 1
                print("โก๏ธ WEIGHT CROSS:{}".format(WEIGHT_CROSS))
        except:
            pass

        print(type(WEIGHT_CROSS))

        
        "ในใใฌในใฎใฟใงๆปใๅ ดๆๆฑบๅฎใใๅ ดๅ"
        # try:
        #     w = w.tolist()
        # except:
        #     pass
        # next_position = BPLIST[w.index(min(w))]
        "----------------------------------------------------------------------"
        "ในใใฌใน+็งปๅใณในใใงๆปใๅ ดๆใๆฑบๅฎใใๅ ดๅ"
        next_position = BPLIST[WEIGHT_CROSS.index(min(WEIGHT_CROSS))] # stress + cost
        "----------------------------------------------------------------------"




        "----- Add 1114 -----"
        # next_position = pd.Series(next_position, index=self.Node_l)

        return next_position

    # def back_end(self, BPLIST, next_position, w, OBS):
    def back_end(self, BPLIST, next_position, w, OBS, test_index, move_cost_result):
        print(BPLIST)
        
        # bpindex = BPLIST.index(next_position) # comment out 1114


        # Arc = [(abs(BPLIST[bpindex].row-BPLIST[x].row)) for x in range(len(BPLIST))]
        Arc = [math.sqrt((BPLIST[-1].row - BPLIST[x].row) ** 2 + (BPLIST[-1].column - BPLIST[x].column) ** 2) for x in range(len(BPLIST))]




        
        print("๐ Arc[็งปๅใณในใ]:{}".format(Arc))
        # index = Arc.index(0)
        # Arc.pop(index)
        print("๐ Arc(remove 0[็พๅจไฝ็ฝฎ]):{}".format(Arc))
        print("๐ Storage {}".format(BPLIST))
        # BPLIST.remove(next_position)
        print("๐ Storage(remove) {}".format(BPLIST))
        # w = np.delete(w, bpindex)

        w = BPLIST
        print("๐ฅ WEIGHT(remove):{}".format(w))

        # print("๐ฅ OBS:{}".format(OBS))
        # OBS = np.delete(OBS, bpindex)
        try:
            # OBS.pop(bpindex)
            OBS.pop(test_index)
        except:
            OBS = OBS.tolist()
            # OBS.pop(bpindex)
            OBS.pop(test_index)
        print("๐ฅ OBS(remove):{}".format(OBS))

        return BPLIST, w, Arc, OBS

    def next_direction_decision(self, trigar__or__reverse):
        if self.Advance_action == self.actions[0]: # Action.UP:
            self.BP_action = self.actions[1] # [0]
            next_diretion_trigar = [(self.actions[1]), (self.actions[0]), (self.actions[2]), (self.actions[3])]
            next_diretion_trigar_reverse = [(self.actions[0]), (self.actions[1]), (self.actions[2]), (self.actions[3])]
        elif self.Advance_action == self.actions[1]: # Action.DOWN:
            self.BP_action = self.actions[0] # [1]
            next_diretion_trigar = [(self.actions[0]), (self.actions[1]), (self.actions[2]), (self.actions[3])]
            next_diretion_trigar_reverse = [(self.actions[1]), (self.actions[0]), (self.actions[2]), (self.actions[3])]
        elif self.Advance_action == self.actions[2]: # Action.LEFT:
            self.BP_action = self.actions[3] # [2]
            next_diretion_trigar = [(self.actions[3]), (self.actions[2]), (self.actions[0]), (self.actions[1])]
            next_diretion_trigar_reverse = [(self.actions[2]), (self.actions[3]), (self.actions[0]), (self.actions[1])]
        elif self.Advance_action == self.actions[3]: # Action.RIGHT:
            self.BP_action = self.actions[2] # [3]
            next_diretion_trigar = [(self.actions[2]), (self.actions[3]), (self.actions[0]), (self.actions[1])]
            next_diretion_trigar_reverse = [(self.actions[3]), (self.actions[2]), (self.actions[0]), (self.actions[1])]
        else:
            next_diretion_trigar, next_diretion_trigar_reverse = self.next_direction_decision_prev_action()

        if trigar__or__reverse == "trigar":
            print("tigar__or__reverse : {}".format(trigar__or__reverse))
            return next_diretion_trigar
        if trigar__or__reverse == "reverse":
            print("tigar__or__reverse : {}".format(trigar__or__reverse))
            return next_diretion_trigar_reverse

    def next_direction_decision_prev_action(self):
        if self.prev_action == self.actions[0]: # Action.UP:
            self.BP_action = self.actions[1]
            next_diretion_trigar = [(self.actions[1]), (self.actions[0]), (self.actions[2]), (self.actions[3])]
            next_diretion_trigar_reverse = [(self.actions[0]), (self.actions[1]), (self.actions[2]), (self.actions[3])]
        elif self.prev_action == self.actions[1]: # Action.DOWN:
            self.BP_action = self.actions[0]
            next_diretion_trigar = [(self.actions[0]), (self.actions[1]), (self.actions[2]), (self.actions[3])]
            next_diretion_trigar_reverse = [(self.actions[1]), (self.actions[0]), (self.actions[2]), (self.actions[3])]
        elif self.prev_action == self.actions[2]: # Action.LEFT:
            self.BP_action = self.actions[3]
            next_diretion_trigar = [(self.actions[3]), (self.actions[2]), (self.actions[0]), (self.actions[1])]
            next_diretion_trigar_reverse = [(self.actions[2]), (self.actions[3]), (self.actions[0]), (self.actions[1])]
        elif self.prev_action == self.actions[3]: # Action.RIGHT:
            self.BP_action = self.actions[2]
            next_diretion_trigar = [(self.actions[2]), (self.actions[3]), (self.actions[0]), (self.actions[1])]
            next_diretion_trigar_reverse = [(self.actions[3]), (self.actions[2]), (self.actions[0]), (self.actions[1])]

        return next_diretion_trigar, next_diretion_trigar_reverse

    def advance_direction_decision(self, dir):

        test = random.sample(dir, len(dir))
        print("test dir : {}, dir : {}".format(test, dir))
        # test = [(self.actions[3]), (self.actions[1]), (self.actions[0]), (self.actions[1])]
        # test = [(self.actions[2]), (self.actions[1]), (self.actions[3]), (self.actions[0])]
        return test # random.shuffle(dir)
        #  [<Action.RIGHT: -2>, <Action.DOWN: -1>, <Action.UP: 1>, <Action.LEFT: 2>]
