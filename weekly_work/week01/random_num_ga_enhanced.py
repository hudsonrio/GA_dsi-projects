from random import randint
from __future__ import print_function

# Tell the user what they are about to experience
print "Are you excited?  It's time to play guess that number *ENHANCED EDITION*!"

# In a notebook, this will prompt the user to enter a number, and pause execution of the app.
# We use int() because anything that comes into
def play_guess_that_number(answer, chances_taken):
    answer = randint(1,5)
    chances_taken = 0
    chances_remaining = 5 - chances_taken

    while chances_remaining >0:
        try:
            response = int(raw_input("Please guess a whole number that is 5 or less. If you are bored and want to stop, say 'stop'"))
            if response > answer:
                print ("too high")
                chances_taken = chances_taken + 1
                guess_number(answer)
            elif response < answer:
                print ("too low")
                chances_taken = chances_taken + 1
                guess_number(answer)
            elif response == answer:
                print ("You're the winner. Let's play again!")
                chances_taken = chances_taken + 1
                guess_number(answer)
            elif response == 'stop':
                break
        except ValueError:
            print("I did not understand that. Please guess a number from 1 through 5. \n You may not spell out your number")
            guess_number(answer)


# We pass "random_number" so it can be referenced throughout
play_guess_that_number(answer, chances_taken)
