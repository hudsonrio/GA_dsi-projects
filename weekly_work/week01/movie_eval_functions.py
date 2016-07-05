# Dictionary of movies

movies = [
{
"name": "Usual Suspects",
"imdb": 7.0,
"category": "Thriller"
},
{
"name": "Hitman",
"imdb": 6.3,
"category": "Action"
},
{
"name": "Dark Knight",
"imdb": 9.0,
"category": "Adventure"
},
{
"name": "The Help",
"imdb": 8.0,
"category": "Drama"
},
{
"name": "The Choice",
"imdb": 6.2,
"category": "Romance"
},
{
"name": "Colonia",
"imdb": 7.4,
"category": "Romance"
},
{
"name": "Love",
"imdb": 6.0,
"category": "Romance"
},
{
"name": "Bride Wars",
"imdb": 5.4,
"category": "Romance"
},
{
"name": "AlphaJet",
"imdb": 3.2,
"category": "War"
},
{
"name": "Ringing Crime",
"imdb": 4.0,
"category": "Crime"
},
{
"name": "Joking muck",
"imdb": 7.2,
"category": "Comedy"
},
{
"name": "What is the name",
"imdb": 9.2,
"category": "Suspense"
},
{
"name": "Detective",
"imdb": 7.0,
"category": "Suspense"
},
{
"name": "Exam",
"imdb": 4.2,
"category": "Thriller"
},
{
"name": "We Two",
"imdb": 7.2,
"category": "Romance"
}
]

# Write a function that takes a single movie and
# returns True if its IMDB score is above 5.5

def simple_movie_eval(user_response):
    user_response == raw_input("What is the movie title? \n")
    for i in movies:
        if i["name"] == user_response:
            score = i["imdb"]
            if score >= 5.5:
                return True
            else:
                return False

#simple_movie_eval("We Two")


def list_movie_eval():
    good_movies = []
    for i in movies:
        if i["imdb"] >= 5.5:
            good_movies.append(i["name"])
    print (good_movies)

#list_movie_eval()

def list_movies_category():
    fav_genre = raw_input("What is your preferred category? ")
    categorized_movies = []
    for i in movies:
        if i["category"] == fav_genre:
            categorized_movies.append(i["name"])
    print (categorized_movies)

#list_movies_category()


# Write a function that takes a list of movies and computes
# the average IMDB score.

def average_imdb():
    total_score = 0
    num_movies = 0
    for i in movies:
        total_score = total_score + i["imdb"]
        num_movies = num_movies + 1
    return total_score/num_movies

#average_imdb()


# Write a function that takes a category and computes
# the average IMDB score (HINT: reuse the function
# from question 2.)


def eval_category():
    fav_genre = raw_input("What is your preferred category? ")
    categorized_movies = []
    total_score = 0
    num_movies = 0
    for i in movies:
        if i["category"] == fav_genre:
            categorized_movies.append(i["name"])
            total_score = total_score + i["imdb"]
            num_movies = num_movies + 1
    return total_score/num_movies

#eval_category()

#input_list = ["The Help", "Bride Wars"]

def average_imdb_in_list():
    total_score = 0
    num_movies = 0
    for i in movies:
        if i["name"] in input_list:
            total_score = total_score + i["imdb"]
            num_movies = num_movies + 1
    return total_score/num_movies

average_imdb_in_list()

input_list = ["The Help", "Bride Wars"]

def average_imdb_in_list():
    total_score = 0
    num_movies = 0
    for i in movies:
        if i["name"] in input_list:
            total_score = total_score + i["imdb"]
            num_movies = num_movies + 1
    return total_score/num_movies

average_imdb_in_list()
