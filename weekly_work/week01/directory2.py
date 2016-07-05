

phonebook = { "Jerry" : "212-555-3015",
              "Elaine" : "212-683-5555",
              "Kramer" : "212-555-0804",
              "George" : "646-111-0000",
              "Newman" : "917-666-6666"
              }

def call_a_friend():
    answer = raw_input("Who do you want to call?\n")
    if answer in phonebook:
        print ("You're calling {} at this number: {}".format(answer, phonebook.get(answer)))
    else:
        print ("Sorry I do not know that person!")

call_a_friend()
