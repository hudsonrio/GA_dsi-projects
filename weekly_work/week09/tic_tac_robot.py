import random
import itertools

unclaimed_spaces = [1,2,3,4,5,6,7,8,9]

def winner(lista, listb):
    win_combos = [[1,4,7], [2,5,8], [3,6,9], [1,2,3], [4,5,6], [7,8,9], [1,5,9], [7,5,3]]
    if len(lista)>=3:
        for h in win_combos:
            if h[0] in listb:
                if h[1] in listb:
                    if h[2] in listb:
                        return('b')
            if h[0] in lista:
                if h[1] in lista:
                    if h[2] in lista:
                        return('a')


def pick_tic(unclaimed_spaces):
    lista = []
    listb = []
    while len(unclaimed_spaces)> 0:#while (len(lista)+len(listb)) < 9:
        if len(lista) <= len(listb): #this is because if player b enters the wrong thing, player a can pick twice
            picka = raw_input("Please select a number 1-9, corresponding to a 3x3 tic tac toe board. (1 is top left, 5 is middle, 9 is bottom right)")
            picka = int(picka)
            if picka in unclaimed_spaces:
                lista.append(picka)
                unclaimed_spaces.remove(picka)
                print("The remaining spaces are:", unclaimed_spaces, "Player A has selected:", lista, "Player B has selected:", listb)
                if winner(lista, listb) == 'a':
                    print("Player A Wins! Victory!")
                    break
                elif winner(lista, listb) == 'b':
                    print("Player B Wins! Victory!")
                    break
                pickb = playerb_strategy(unclaimed_spaces)
                pickb = int(pickb)
                if pickb in unclaimed_spaces:
                    listb.append(pickb)
                    unclaimed_spaces.remove(pickb)
                    print("The remaining spaces are:", unclaimed_spaces, "Player A has selected:", lista, "Player B has selected:", listb)
                    if winner(lista, listb) == 'a':
                        print("Player A Wins! Victory!")
                        break
                    elif winner(lista, listb) == 'b':
                        print("Player B Wins! Victory!")
                        break
                else:
                    print("That number is not available - or you picked something that does not make sense.")
                    continue
            else:
                print("That number is not available - or you picked something that does not make sense.")
                continue
    return("Its a tie! Nobody won. Play again!")


def playerb_strategy(unclaimed_spaces):
    tier1=5
    tier2 = [2,4,6,8]
    tier3= [1,7,3,9]
    if 5 in unclaimed_spaces:
        return 5
    for i in itertools.permutations(tier2,1):
        try:
            if i in unclaimed_spaces:
                return(i)
        except:
            for j in intertools.permutations(tier3,1):
                if j in unclaimed_spaces:
                    return(j)




pick_tic(unclaimed_spaces)
