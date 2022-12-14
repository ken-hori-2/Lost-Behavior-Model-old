
import math
from reference_match_rate import Property

class Algorithm_bp():

    def __init__(self, *arg):
        
        self.state = arg[0] # state
        self.env = arg[1] # env
        self.agent = arg[2] # agent
        self.NODELIST = arg[3] # NODELIST
        self.Observation = arg[4]
        self.refer = Property()
        # self.Cal = Cal()
        ########## parameter ##########
        self.total_stress = 0
        self.stress = 0
        self.Stressfull = 8 # 10 # 4
        self.COUNT = 0
        self.done = False
        self.TRIGAR = False
        self.TRIGAR_REVERSE = False
        self.BACK = False
        self.BACK_REVERSE = False
        self.on_the_way = False
        self.bf = True
        ########## parameter ##########
        self.STATE_HISTORY = []
        self.BPLIST = []
        self.PROB = []
        self.Arc = []
        self.OBS = []
        # self.next_position = []
        self.Storage_Arc = []
        self.SAVE = []
            

    
    def BP(self, STATE_HISTORY, state, TRIGAR, OBS, BPLIST, action, Add_Advance, total_stress, SAVE_ARC, TOTAL_STRESS_LIST, Node_s, Node_A, Node_B, Node_C, Node_D, Node_g, Cost_S, Cost_O, Cost_A, Cost_B, Cost_C, Cost_D, WEIGHT_CROSS_S, WEIGHT_CROSS_O, WEIGHT_CROSS_A, WEIGHT_CROSS_B, WEIGHT_CROSS_C, WEIGHT_CROSS_D):
        self.STATE_HISTORY = STATE_HISTORY
        self.state = state
        self.TRIGAR = TRIGAR
        self.OBS = OBS
        self.BPLIST = BPLIST
        self.Advance_action = action
        self.bf = True
        self.state_history_first = True
        self.Add_Advance = Add_Advance
        self.total_stress = total_stress
        self.SAVE_ARC = SAVE_ARC
        self.first_pop = True
        self.BackPosition_finish = False
        pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()
        self.TOTAL_STRESS_LIST = TOTAL_STRESS_LIST

        "============================================== Robosin ver. との違い =============================================="
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
        "============================================== Robosin ver. との違い =============================================="


        
        while not self.done:
        
            print("\n-----{}Steps-----".format(self.COUNT+1))

            if self.COUNT > 30:
                print("\n############################# test ####################################\n")
                if self.NODELIST[self.state.row][self.state.column] in pre:
                    print("\n############################# test ####################################\n")
                    print("\n============================\n🤖 🔛　アルゴリズム切り替え\n============================")
                    break
            if self.TRIGAR_REVERSE:
                if self.BACK_REVERSE:
                    try:
                        
                        print(f"🥌 WEIGHT = {self.w}")
                        print("👟 Arc[移動コスト]:{}".format(self.Arc))

                        print("📂 Storage {}".format(self.BPLIST))
                        
                        # callback
                        self.next_position, WEIGHT_CROSS = self.agent.back_position(self.BPLIST, self.w, self.Arc)
                        # self.next_position, w, Arc, WEIGHT_CROSS = self.agent.back_position(self.BPLIST, self.w, self.Arc)


                        print(f"========Decision Next State=======\n⚠️  NEXT POSITION:{self.next_position}\n==================================")
                        self.on_the_way = True
                        

                        self.BACK_REVERSE = False


                        "--test--"
                        CS = 0
                        CA = 0
                        CB = 0
                        CC = 0
                        CD = 0
                        CO = 0
                        WCS = 0
                        WCA = 0
                        WCB = 0
                        WCC = 0
                        WCD = 0
                        WCO = 0
                        # self.Cost_S.append(0)
                        for i in range(len(self.Arc)):
                            if i == 0:
                                CO = self.Arc[i] 
                                # WCO = WEIGHT_CROSS[i]
                            elif i == 1:
                                CA = self.Arc[i]
                                # WCA = WEIGHT_CROSS[i]
                            elif i == 2:
                                CB = self.Arc[i] 
                                # WCB = WEIGHT_CROSS[i]
                            elif i == 3:
                                CC = self.Arc[i] 
                                # WCC = WEIGHT_CROSS[i]
                            elif i == 4:
                                CD = self.Arc[i]
                                # WCD = WEIGHT_CROSS[i]

                        for i in range(len(WEIGHT_CROSS)):
                            if i == 0:
                                # CO = self.Arc[i] 
                                WCO = WEIGHT_CROSS[i]
                            elif i == 1:
                                # CA = self.Arc[i]
                                WCA = WEIGHT_CROSS[i]
                            elif i == 2:
                                # CB = self.Arc[i] 
                                WCB = WEIGHT_CROSS[i]
                            elif i == 3:
                                # CC = self.Arc[i] 
                                WCC = WEIGHT_CROSS[i] 
                            elif i == 4:
                                # CD = self.Arc[i]
                                WCD = WEIGHT_CROSS[i]

                        self.Cost_S.append(CS)
                        self.Cost_A.append(CA)
                        self.Cost_B.append(CB)
                        self.Cost_C.append(CC)
                        self.Cost_D.append(CD)
                        self.Cost_O.append(CO)

                        self.WEIGHT_CROSS_S.append(WCS)
                        self.WEIGHT_CROSS_O.append(WCO)
                        self.WEIGHT_CROSS_A.append(WCA)
                        self.WEIGHT_CROSS_B.append(WCB)
                        self.WEIGHT_CROSS_C.append(WCC)
                        self.WEIGHT_CROSS_D.append(WCD)

                        # self.Cost_S.append(Arc)
                        # self.Cost_O.append(WEIGHT_CROSS)
                    except:
                    # except Exception as e:
                    #     print('=== エラー内容 ===')
                    #     print('type:' + str(type(e)))
                    #     print('args:' + str(e.args))
                    #     print('message:' + e.message)
                    #     print('e自身:' + str(e))
                        print("ERROR!")
                        # self.STATE_HISTORY.append(self.state)

                        self.BackPosition_finish = True
                        break
                
                
                if int(self.state.row) < int(self.next_position.row):
                        self.TRIGAR_REVERSE = False
                elif int(self.state.column) > int(self.next_position.column):
                        self.TRIGAR_REVERSE = False

                
                    
                try:
                
                    if self.state == self.next_position:

                        # callback
                        self.BPLIST, self.w, self.Arc, self.OBS = self.agent.back_end(self.BPLIST, self.next_position, self.w, self.OBS)
                        self.BACK_REVERSE =True
                        print("🔚 ARRIVE AT BACK POSITION (戻り終わりました。)")
                        print(f"🤖 State:{self.state}")

                        
                        # self.total_stress = 0
                        "-- 追加部分 --"
                        delta_s = self.Observation[self.state.row][self.state.column]
                        self.total_stress -= (1-delta_s)
                        "-- 追加部分 --"

                        # Add 1025
                        # self.TOTAL_STRESS_LIST.append(self.total_stress)
                        # self.TOTAL_STRESS_LIST.append(10)

                        self.STATE_HISTORY.append(self.state)
                        self.TOTAL_STRESS_LIST.append(self.total_stress)
                        # self.TOTAL_STRESS_LIST.append(abs(1.0-self.total_stress))

                        self.Node_s.append(0)
                        self.Node_A.append(0)
                        self.Node_B.append(0)
                        self.Node_C.append(0)
                        self.Node_D.append(0)
                        self.Node_g.append(0)

                        "--test--"
                        self.Cost_S.append(CS)
                        self.Cost_A.append(CA)
                        self.Cost_B.append(CB)
                        self.Cost_C.append(CC)
                        self.Cost_D.append(CD)
                        self.Cost_O.append(CO)

                        self.WEIGHT_CROSS_S.append(WCS)
                        self.WEIGHT_CROSS_O.append(WCO)
                        self.WEIGHT_CROSS_A.append(WCA)
                        self.WEIGHT_CROSS_B.append(WCB)
                        self.WEIGHT_CROSS_C.append(WCC)
                        self.WEIGHT_CROSS_D.append(WCD)
                        
                        
                        # 0921 統合テスト
                        print("\n============================\n🤖 🔛　アルゴリズム切り替え\n============================")
                        break
                        
                except:
                # except Exception as e:
                #     print('=== エラー内容 ===')
                #     print('type:' + str(type(e)))
                #     print('args:' + str(e.args))
                #     print('message:' + e.message)
                #     print('e自身:' + str(e))
                    print("state:{}".format(self.state))
                    print("これ以上戻れません。 終了します。")
                    
                    # if self.NODELIST[self.prev_state.row][self.prev_state.column] == 0: # > 0.0:
                    #     if self.total_stress + self.stress >= 0:
                    #         self.total_stress += self.stress
                    break

            

            
            if self.BACK or self.bf:
                    try:
                        
                        if self.bf: # ストレスが溜まってから初回
                            self.w = self.OBS
                            print(f"🥌 WEIGHT = {self.w}")
                            print("SAVE ARC : {}".format(self.SAVE_ARC))
                            
                            if self.Add_Advance:
                                # ユークリッド距離
                                self.Arc = [math.sqrt((self.BPLIST[-1].row - self.BPLIST[x].row) ** 2 + (self.BPLIST[-1].column - self.BPLIST[x].column) ** 2) for x in range(len(self.BPLIST))]


                                print("👟 Arc[移動コスト]:{}".format(self.Arc))
                                index = self.Arc.index(0)
                                self.Arc.pop(index)
                                print("👟 Arc(remove 0[現在位置]):{}".format(self.Arc))
                                print("📂 Storage {}".format(self.BPLIST))


                                # if self.Add_Advance:
                                self.BPLIST.pop(-1) # advanceアドバンスで追加した現在地の文を削除
                                # でも、advanceで追加してない時は消しちゃいけない
                                # おそらくアークも消してしまっている？？

                                # self.SAVE_ARC.pop(0)

                                # self.Storage_Arc = self.Cal.caluculate(self.SAVE_ARC)
                                # print("Storage Arc : {}".format(self.Storage_Arc))

                            
                                print("📂 Storage(remove) {}".format(self.BPLIST))

                            print("👟 Arc[移動コスト]:{}".format(self.Arc))
                            print("👟 Arc(remove 0[現在位置]):{}".format(self.Arc))
                            print("📂 Storage {}".format(self.BPLIST))

                            
                        else:
                            print(f"🥌 WEIGHT = {self.w}")
                            print("👟 Arc[移動コスト]:{}".format(self.Arc))

                            print("📂 Storage {}".format(self.BPLIST))
                        self.bf = False
                        self.BACK = False
                        
                        # callback
                        self.next_position, WEIGHT_CROSS = self.agent.back_position(self.BPLIST, self.w, self.Arc)

                        # self.next_position, w, Arc, WEIGHT_CROSS = self.agent.back_position(self.BPLIST, self.w, self.Arc)


                        print(f"========Decision Next State=======\n⚠️  NEXT POSITION:{self.next_position}\n==================================")
                        self.on_the_way = True


                        "--test--"
                        # try:
                        CS = 0
                        CA = 0
                        CB = 0
                        CC = 0
                        CD = 0
                        CO = 0
                        WCS = 0
                        WCA = 0
                        WCB = 0
                        WCC = 0
                        WCD = 0
                        WCO = 0
                        # self.Cost_S.append(0)
                        for i in range(len(self.Arc)):
                            if i == 0:
                                CO = self.Arc[i] 
                                # WCO = WEIGHT_CROSS[i] 
                            elif i == 1:
                                CA = self.Arc[i]
                                # WCA = WEIGHT_CROSS[i]
                            elif i == 2:
                                CB = self.Arc[i]
                                # WCB = WEIGHT_CROSS[i]
                            elif i == 3:
                                CC = self.Arc[i] 
                                # WCC = WEIGHT_CROSS[i]
                            elif i == 4:
                                CD = self.Arc[i]
                                # WCD = WEIGHT_CROSS[i]

                        for i in range(len(WEIGHT_CROSS)):
                            if i == 0:
                                # CO = self.Arc[i] 
                                WCO = WEIGHT_CROSS[i]
                            elif i == 1:
                                # CA = self.Arc[i]
                                WCA = WEIGHT_CROSS[i]
                            elif i == 2:
                                # CB = self.Arc[i] 
                                WCB = WEIGHT_CROSS[i]
                            elif i == 3:
                                # CC = self.Arc[i] 
                                WCC = WEIGHT_CROSS[i] 
                            elif i == 4:
                                # CD = self.Arc[i]
                                WCD = WEIGHT_CROSS[i]
                        # except:
                        #     print("error")
                        "-- Add 1031 --"
                        self.Cost_S.append(CS)
                        self.Cost_A.append(CA)
                        self.Cost_B.append(CB)
                        self.Cost_C.append(CC)
                        self.Cost_D.append(CD)
                        self.Cost_O.append(CO)

                        self.WEIGHT_CROSS_S.append(WCS)
                        self.WEIGHT_CROSS_O.append(WCO)
                        self.WEIGHT_CROSS_A.append(WCA)
                        self.WEIGHT_CROSS_B.append(WCB)
                        self.WEIGHT_CROSS_C.append(WCC)
                        self.WEIGHT_CROSS_D.append(WCD)
                            
                        # self.Cost_S.append(Arc)
                        # self.Cost_O.append(WEIGHT_CROSS)

                        
                    except:
                    # except Exception as e:
                    #     print('=== エラー内容 ===')
                    #     print('type:' + str(type(e)))
                    #     print('args:' + str(e.args))
                    #     print('message:' + e.message)
                    #     print('e自身:' + str(e))
                        print("ERROR!")
                        # self.STATE_HISTORY.append(self.state)

                        print("リトライ行動終了！")


                        print(" = 戻り切った状態 🤖🔚")
                        self.BackPosition_finish = True
                        
                        break

            if int(self.state.row) > int(self.next_position.row):
                self.TRIGAR_REVERSE = True
            elif int(self.state.column) < int(self.next_position.column):
                self.TRIGAR_REVERSE = True
                
            try:

                    if self.state == self.next_position:

                        # self.lost = False

                        self.BPLIST, self.w, self.Arc, self.OBS = self.agent.back_end(self.BPLIST, self.next_position, self.w, self.OBS)
                        self.BACK =True
                        print("🔚 ARRIVE AT BACK POSITION (戻り終わりました。)")
                        print(f"🤖 State:{self.state}")
                        print("OBS : {}".format(self.OBS))

                        # self.total_stress = 0
                        "-- 追加部分 --"
                        print("⚠️ total : {}".format(self.total_stress))
                        delta_s = self.Observation[self.state.row][self.state.column]
                        self.total_stress -= (1-delta_s)
                        print("⚠️ delta_s : {}".format(delta_s))
                        print("⚠️ total : {}".format(self.total_stress))
                        "-- 追加部分 --"
                        "============================================== Robosin ver. との違い =============================================="

                        self.STATE_HISTORY.append(self.state)
                        self.TOTAL_STRESS_LIST.append(self.total_stress)
                        # self.TOTAL_STRESS_LIST.append(abs(1.0-self.total_stress))

                        self.Node_s.append(0)
                        self.Node_A.append(0)
                        self.Node_B.append(0)
                        self.Node_C.append(0)
                        self.Node_D.append(0)
                        self.Node_g.append(0)

                        # "--test--"
                        # self.Cost_S.append(0)
                        # self.Cost_A.append(0)
                        # self.Cost_B.append(0)
                        # self.Cost_C.append(0)
                        # self.Cost_D.append(0)
                        # self.Cost_O.append(0)

                        "-- Add 1031 --"
                        self.Cost_S.append(CS)
                        self.Cost_A.append(CA)
                        self.Cost_B.append(CB)
                        self.Cost_C.append(CC)
                        self.Cost_D.append(CD)
                        self.Cost_O.append(CO)

                        self.WEIGHT_CROSS_S.append(WCS)
                        self.WEIGHT_CROSS_O.append(WCO)
                        self.WEIGHT_CROSS_A.append(WCA)
                        self.WEIGHT_CROSS_B.append(WCB)
                        self.WEIGHT_CROSS_C.append(WCC)
                        self.WEIGHT_CROSS_D.append(WCD)
                        "============================================== Robosin ver. との違い =============================================="
                        


                        # 0921 統合テスト
                        print("\n============================\n🤖 🔛　アルゴリズム切り替え\n============================")
                        break

                        COUNT += 1
                        continue
                    else:   
                        if self.on_the_way:
                            self.on_the_way = False
                        else:
                            print("🔛 On the way BACK") 
            except:
                # except Exception as e:
                #     print('=== エラー内容 ===')
                #     print('type:' + str(type(e)))
                #     print('args:' + str(e.args))
                #     print('message:' + e.message)
                #     print('e自身:' + str(e))
                    print("state:{}".format(self.state))
                    print("これ以上戻れません。 終了します。")
                    break # expansion 無しの場合は何回も繰り返さない
                
            print(f"🤖 State:{self.state}")
            if not self.state_history_first:
                self.STATE_HISTORY.append(self.state) # これがないと戻る行動が可視化されない
                self.TOTAL_STRESS_LIST.append(self.total_stress)
                # self.TOTAL_STRESS_LIST.append(abs(1.0-self.total_stress))

                "============================================== Robosin ver. との違い =============================================="
                self.Node_s.append(0)
                self.Node_A.append(0)
                self.Node_B.append(0)
                self.Node_C.append(0)
                self.Node_D.append(0)
                self.Node_g.append(0)

                self.Cost_S.append(CS) #0)
                self.Cost_A.append(CA) #0)
                self.Cost_B.append(CB) #0)
                self.Cost_C.append(CC) #0)
                self.Cost_D.append(CD) #0)
                self.Cost_O.append(CO) #0)

                self.WEIGHT_CROSS_S.append(WCS)
                self.WEIGHT_CROSS_O.append(WCO)
                self.WEIGHT_CROSS_A.append(WCA)
                self.WEIGHT_CROSS_B.append(WCB)
                self.WEIGHT_CROSS_C.append(WCC)
                self.WEIGHT_CROSS_D.append(WCD)
                "============================================== Robosin ver. との違い =============================================="
                
            print(f"Total Stress:{self.total_stress}")
            print("TRIGAR : {}".format(self.TRIGAR))
            self.state_history_first = False


            # self.state = self.next_position # 戻る行動を省略したver.
            
            self.action, self.Reverse   , self.lost = self.agent.policy_bp(self.state, self.TRIGAR, self.TRIGAR_REVERSE, self.COUNT)

            # self.next_state, self.stress, self.done = self.env.step(self.action, self.TRIGAR)

            # self.next_state, self.stress, self.done = self.env._move(self.state, self.action, self.TRIGAR, All_explore, self.Reverse)
            self.next_state, self.stress, self.done = self.env.step(self.state, self.action, self.TRIGAR)
            # self.prev_state = self.state # 1つ前のステップを保存 -> 後でストレスの減少に使う
            self.state = self.next_state
            
            
            print("COUNT : {}".format(self.COUNT))
            if self.COUNT > 100:
                print("\n######## BREAK ########\n")
                # breakではなくて、戻る場所に戻れないから別の戻る場所にするとか
                print("\n📂 Storage {}\n\n\n".format(self.BPLIST))
                break
            self.COUNT += 1
        self.COUNT = 0

        return self.total_stress, self.STATE_HISTORY, self.state, self.OBS, self.BackPosition_finish, self.TOTAL_STRESS_LIST, self.Node_s, self.Node_A, self.Node_B, self.Node_C, self.Node_D, self.Node_g, self.Cost_S, self.Cost_O, self.Cost_A, self.Cost_B, self.Cost_C, self.Cost_D, self.WEIGHT_CROSS_S, self.WEIGHT_CROSS_O, self.WEIGHT_CROSS_A, self.WEIGHT_CROSS_B, self.WEIGHT_CROSS_C, self.WEIGHT_CROSS_D