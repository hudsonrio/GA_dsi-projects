weight1 = float(input("How many pounds does your suitcase weigh?"))
if weight1 % 2 == 0:
    print ("I don't understand even numbers")
else:
    if weight1 > 50:
        print ("There is a $25 charge for luggage that heavy")
print ("Thank you for your tips!")
