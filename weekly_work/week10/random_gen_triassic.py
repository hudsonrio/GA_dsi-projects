import itertools
import random


def random_triassic():
    students = ['Michael', 'Kelly', 'Tim', 'Phillippa', 'Peter', 'Katty', 'Rebecca', 'Tamara', 'Tucker', 'Hudson', 'Lydia', 'Sam']
    try:
        group_size = int(raw_input("Welcome to the triassic randomizer! How big do you want your groups to be? (e.g. type 1 for randomized list of students, or 2 for pairs). If you enter a number greater than 6, it will return a schedule of random pairings as long as your number."))
        if group_size < 7:
            groups = []
            single_groups = []
            while len(students) >= group_size:
                seq = list(itertools.combinations(students, group_size))
                a = random.choice(seq)
                groups.append(a)
                for b in a:
                    students.remove(b)
                if len(students) < group_size:
                    if len(students)>1:
                        groups.append(students)
            if group_size == 1:
                for c in groups:
                    for d in c:
                        single_groups.append(d)
                print single_groups
            else:
                print groups
        elif group_size >= 7:
            sched_groups = []
            for i in range(group_size):
                students_sched = ['Michael', 'Kelly', 'Tim', 'Phillippa', 'Peter', 'Katty', 'Rebecca', 'Tamara', 'Tucker', 'Hudson', 'Lydia', 'Sam']
                while len(students_sched) > 0:
                    seq = list(itertools.combinations(students_sched, 2))
                    e = random.choice(seq)
                    sched_groups.append(["Day number "+str(i+1), e])
                    for f in e:
                        students_sched.remove(f)
            print sched_groups
    except ValueError:
        print("Please enter a integer number (e.g. 2, not 'two' or 2.0)")
        random_triassic()

random_triassic()
