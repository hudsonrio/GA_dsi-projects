#this is my first attempts at the Catan Project


#Need an entry method for entering each of the 19 spots on the board
'''These should be stored in a dict, which should correspond with the 54 possible buildable
locations on the board. This should also be related to a) its number, b) its resource, and
c) any corresponding ports'''

#planning on making the assumption that the desert is in the middle, so the R2 and R3
#are the only spaces that need to be considered, and the 18 available locations can be mapped
#1:1 with the number tokens

def orient_board():
	print("Does the DESERT (center) have a flat size facing where 'A' was placed? y/n")
	b_orient = input("> ")
	#this is to decide which dataframe to use to value each settlement the player enters later
	#in other words, in  layout 1 and 2, position 15 represents a different space; this helps us choose
	if b_orient.lower() == "y" or "yes":
		#assume layout 1 (https://docs.google.com/spreadsheets/d/1JHsyIii01yk_skgJe03O2d9f1km4xuQDrcHMJybUfNM/edit#gid=608684759)
		end

	elif b_orient.lower() == "n" or "no":
		#assume layout 2
		end


def enter_board():
	positions = {}
	#loop through resource_vals and ask where each is placed and save it in resource_pos dict
	print("Please note, this will not work unless/n * the DESERT is placed in the middle",
		 	"of the board, and you spread the tiles counterclockwise around the board./n/n"

	orient_board()

	for loc, val in resource_vals
	#go through all 18 spots, but no more
    print("\nI can't see the board! What type of resource is the {} tile? Please select from:",
    	 "/n *Grain/n *Wood/n *Brick/n *Sheep/n *Ore/n").format(loc)
    new_stuff = input("> ")

    if new_stuff.lower() == "grain" or "wheat":
        resource_pos.append(loc: "g").
        continue
    elif new_stuff.lower() == "wood" or "lumber":
        resource_pos.append(loc: "w").
        continue
    elif new_stuff.lower() == "brick":
        resource_pos.append(loc: "b").
        continue
	elif new_stuff.lower() == "sheep" or "wool":
        resource_pos.append(loc: "s").
        continue
    elif new_stuff.lower() == "ore" or "iron":
        resource_pos.append(loc: "o").
        continue
    else:
    	print("I did not understand that. Grain, Wood, Brick, Sheep or Ore please")
    	continue


def enter_plcmts():
	positions = {}
	for i,j in range (1,54)
		positions.append(i)
	#this should prompt them to enter where their settlements are, after the board has been fully entered.

	print("\nOkay, now I get what the board looks like. What spot are you thinking about? Please enter the number, from 1-54")
	settlement1 = int(input("> "))
	if ValueError:
		print ("Please enter a number between 1 and 54 (you cannot spell it out or use decimals)")
	else:
		positions.append(positions.get(settlement1))

	if settlement1 ==

    print("Type DONE to quit, SHOW to see the current list,"
          "REMOVE to delete an item, and HELP to get this message.")
#Created a spreadsheet with two board layouts, with each position mapped to the spaces it touches
#https://docs.google.com/spreadsheets/d/1JHsyIii01yk_skgJe03O2d9f1km4xuQDrcHMJybUfNM/edit#gid=608684759

positions_layout1 = {   1: ['a'],
						2: ['a'],
						3: ['a', 'b'],
						4: ['a', 'b'],
						5: ['b','c'],
						6: ['c'],
						7: ['c'],
						8: ['c','d'],
						9: ['d'],
						10: ['d', 'e'],
						#this is the last accurate location
						11: ['a'],
						12: ['a'],
						13: ['a'],
						14: ['a', 'b'],
						15: ['a', 'b'],
						16: ['a', 'b'],
						17: ['a'],
						18: ['a'],
						19: ['a'],
						20: ['a'],
						21: ['a', 'b'],
						22: ['a', 'b'],
						23: ['a', 'b'],
						24: ['a'],
						25: ['a'],
						26: ['a'],
						27: ['a'],
						28: ['a'],
						29: ['a'],
						30: ['a'],
						31: ['a', 'b'],
						32: ['a', 'b'],
						33: ['a', 'b'],
						34: ['a', 'b'],
						35: ['a', 'b'],
						36: ['a', 'b'],
						37: ['a', 'b'],
						38: ['a', 'b'],
						39: ['a', 'b'],
						40: ['a', 'b'],
						41: ['a', 'b'],
						42: ['a', 'b'],
						43: ['a', 'b'],
						44: ['a', 'b'],
						45: ['a', 'b'],
						46: ['a', 'b'],
						47: ['a', 'b'],
						48: ['a', 'b'],
						49: ['a', 'b'],
						50: ['a', 'b'],
						51: ['a'],
						52: ['a', 'b'],
						53: ['a', 'b'],
						54: ['a', 'b'],
					}

positions_layout2 = {   1: ['a'],
						2: ['a','b'],
						3: ['a', b']


#this dict would continue on through 54, and have another comparable one for positions_layout2
#is there an easier way? Can I somehow import the spreadsheet?


}

resource_vals = {   'a': 5,
					'b': 2,
					'c': 6,
					'd': 3,
					'e': 8,
					'f': 10,
					'g': 9,
					'h': 12,
					'i': 11,
					'j': 4,
					'k': 8,
					'l': 10,
					'm': 9,
					'n': 4,
					'o': 5,
					'p': 6,
					'q': 3,
					'r': 11
				}
dict_re_pos = {}
resource_pos = {}



def income_rate_by_resource():

	for i, j in positions_layout1:
		#call each value in list
		if any j in resource_vals
	#need to make sure this is pulling from correct table
	resource_vals.get(j)
	#take all the letters from the correct positions table
