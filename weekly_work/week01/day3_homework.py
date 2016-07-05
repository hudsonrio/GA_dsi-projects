

from __future__ import print_function
import string
#this creates a blank dict

from __future__ import print_function
import string


def dict_changer(target_dict):
    '''this sets a new_dict equal to the target_dict and deletes the data we no longer want. I do this the inverse way
    (only copy over relevant info) below'''
    print(target_dict)
    new_dict = target_dict
    for i,j in new_dict.items():
            if type(i) != str:
                del new_dict[i]
            elif i[0] not in string.ascii_lowercase:
                del new_dict[i]
            elif i[0] in ('aeiou'):
                new_dict[i] = "vowel"
            else:
                new_dict[i] = "consonant"
    print(new_dict)
    return(new_dict)

dict_changer(input_dict)



def new_dict_changer(target_dict):
    '''this function creates a blank dict and then ports over the relevant data based on conditions. It accomplishes same as dict_changer
    but in another way'''
    print(target_dict)
    newer_dict = {}
    for i,j in target_dict.items():
            if type(i) == str:
                if i[0] in string.ascii_lowercase:
                    if i[0] in ('aeiou'):
                        newer_dict[i] = "vowel"
                    else:
                        newer_dict[i] = "consonant"
                else:
                    continue
            else:
                continue
    print(newer_dict)
    return(newer_dict)

new_dict_changer(input_dict)


from __future__ import print_function
import string
#final version as of 2:50pm 6/8

def super_dict_changer(target_dict, remainder=[]):
    print(target_dict)
    print(remainder)
    newest_dict = {}
    if remainder == []:
        remainder.extend(2)
    for i,j in target_dict.items():
        for x in remainder:
            for h in j:
                newest_dict[h] = (float(h)%float(x))
    print(newest_dict)
    return(newest_dict)

super_dict_changer(test_dict, optional_remainder)
