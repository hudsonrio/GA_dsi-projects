import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import os
#import random

#%matplotlib inline

#os.getcwd()
dat = pd.read_csv("/Users/HudsonCavanagh/dsi-projects/side_projects/bastard_bowl-analysis.csv")
bb = pd.DataFrame(dat)

solutions = {'NBA':'Cavs (+4.5)', '*jon killed by*': 'no one', '*ramsay killed by*': 'His dogs', 'who scores most?':'harrison barnes'}
any_three = {'True': ['Littlefinger shows up', 'The giant dies', 'The giant kills more than 10 people', 'Rickon gets killed','Davos finds out what happened to Shireen', 'A dragon kills someone','The other two dragons show up','Reference to wildfire'],
             'False': ['A walker shows up', 'Jamie saves the stark army', 'Brienne dies', 'sansa kills Littlefinger', 'Ramsay feeds someone to the dogs', 'And kills himself', 'The hound joins the fight', 'arya joins the fight ', 'Sam Tarley makes an appearance', 'Littlefinger turns on Sansa', "We find out Varys' secret mission", 'A Direwolf appears', 'Bran visits the past', 'Bran visits the future', 'Bran visits the battle itself', 'Tormund dies', 'Brienne dies', 'Lady Mormont does something badass', 'Lady Mormont dies', 'Jorah shows up']}


def truther(list):
    wins = [] #append names and scores later
    for r, h in enumerate(bb["What's ur name?"]):
        score = []
        sub_score = {"Abby":3, "Lebo":1, "Ashdaddy":2, "Nathan":2, "Paulie":1, "Hudson":3, "Spencer":2, "Devin":2}
        if str(bb['NBA'][r]) == solutions['NBA']:
            score.append(1)
        if str(bb['*jon killed by*'][r]) == solutions['*jon killed by*']:
            score.append(1)
        if str(bb['*ramsay killed by*'][r]) == solutions['*ramsay killed by*']:
            score.append(1)
        if str(bb['who scores most?'][r]) == solutions['who scores most?']:
            score.append(1)
        # magic = [str(bb["*pick any three*"][r]).split(", ")]
        # for t in magic: #still not working correctly, need to do so manually
        #     if str(t)[1:7] in any_three["True"]:
        #         sub_score.append(1)


        wins.append([h, (np.sum(score) + sub_score[h])])
        win = pd.DataFrame(wins)
#print(win)
        ax = sns.barplot(x = win[0], y = win[1], data = win, palette="GnBu_d")
        ax.set_ylabel("Score (out of 7)")
        ax.set_xlabel("Name (kinda)")
        sns.despine(bottom=True)
        plt.setp(ax, yticks=[0,1,2,3,4,5,6])
        #plt.tight_layout(h_pad=2)
        fig = ax.get_figure()
        fig.savefig('bastard_bowl_winners.jpeg')

truther(bb)
