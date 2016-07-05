from random import randint

def guess_number(answer):
    '''
    Generate random number
    Ask for user input
    Compare user input with generated number
    Check winning conditions
    Output response "too high", "too low", or "You're the winner"
    Repeat until user wins
    '''

    answer = randint(1,5)

    try:
        response = int(raw_input("Please guess a whole number that is 5 or less. If you are bored and want to stop, say 'stop'"))
        if response > answer:
            print ("too high")
            guess_number(answer)
        elif response < answer:
            print ("too low")
            guess_number(answer)
        elif response == answer:
            print ("You're the winner. Let's play again!")
            guess_number(answer)
        elif response == 'stop':
            exit()
    except ValueError:
        print("I did not understand that. Please guess a number from 1 through 5. \n You may not spell out your number")
        guess_number(answer)

guess_number(answer)
