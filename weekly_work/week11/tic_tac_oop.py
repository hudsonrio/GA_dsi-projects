import pandas as pd
import numpy as np

df = pd.DataFrame(np.nan,index =[1,2,3], columns = [1,2,3])

class Board():
    def __init__(self, hour=0, box_size=3, computer=0):
        '''select computer = True to play against AI'''
        self.row_set = box_size
        self.col_set = box_size
        board = pd.DataFrame(np.nan, [range(box_size)], [range(box_size)])


    def turn_taker(self, a_turns=0, b_turns=0, a_is_first=1):
        if a_is_first = 0:
            if a_turns == 0 and b_turns ==0:
                if np.random.randint(1, 2) == 1:
                    self.pick_space
                    a_is_first = True
                    self.a_turns += 1
                elif np.random.randint(1, 2) == 2:
                    self.player_b()
            else:
                a_count = 0
                b_count = 0
                for i in range(1,3)
                    for 'X' in df[i]:
                        a_count +=1
                    for 'O' in df[i]:
                        b_count += 1
                if a_count > b_count:
                    self.player_b()
                elif a_count < b_count:
                    self.pick_space
                elif a_count == b_count:
                    if a_is_first == True:
                      self.pick_space # need better course of action if there's a tie - remember who went first
                    else:
                      self.player_b()

    def check_spot(self, ac, bc):
        if df.loc[int(ac),int(bc)] in ['X' , 'O']:
            return False
        elif df.loc[int(ac),int(bc)] not in ['X' , 'O']:
            return True

    def pick_space(self):
        if df.isnull().any().any() == True:
            print 'Your Turn! '
            a = raw_input('which row: ')
            b = raw_input('which column: ')
            if check_spot(a,b) == True:
                df.loc[int(a),int(b)] == 'X'
                print df
                print ' '
                print 'Computer\'s Turn'

    def pick_num_b(self)
        ac = np.random.randint(1, 3)
        bc = np.random.randint(1, 3)
        return ac,bc

        def status_check(self):
        #still workig on this

        unclaimed_spaces.remove(picka)
                print("The remaining spaces are:", unclaimed_spaces, "Player A has selected:", lista, "Player B has selected:", listb)
                if winner(lista, listb) == 'a':
                    print("Player A Wins! Victory!")
                    break
                elif winner(lista, listb) == 'b':
                    print("Player B Wins! Victory!")
                    break



    def player_b(self):
        if df.loc[2,2] != 'X' and df.loc[2,2] != 'O':
            ac, b2 = 2,2
            df.loc[int(ac),int(bc)] = 'O'
            print 'computer played: ' + str(ac) +' '+ str(bc)
        else:
            if check_spot(pick_num_b) == True:
                df.loc[int(ac),int(bc)] = 'O'
