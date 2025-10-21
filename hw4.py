##### Try to use map and reduce in the next 3 exercises
# 1)
# Create a function called "count_simba" that counts and returns
# the number of times that Simba appears in a list of
# strings. Example:
# ["Simba and Nala are lions.", "I laugh in the face of danger.",
#  "Hakuna matata", "Timon, Pumba and Simba are friends, but Simba could eat the other two."] 
from functools import reduce

def count_simba(sentences):
    counts = map(lambda s: s.count("Simba"), sentences)
    total = reduce(lambda a, b: a + b, counts, 0)
    return total

#example
sentences = [
    "Simba and Nala are lions.",
    "I laugh in the face of danger.",
    "Hakuna matata",
    "Timon, Pumba and Simba are friends, but Simba could eat the other two."
]

print(count_simba(sentences))  


# 2)
# Create a function called "get_day_month_year" that takes 
# a list of datetimes.date and returns a pandas dataframe
# with 3 columns (day, month, year) in which each of the rows
# is an element of the input list and has as value its 
# day, month, and year.
# 
import pandas as pd
from functools import reduce

def get_day_month_year(date_list):
    mapped = map(lambda d: {"day": d.day, "month": d.month, "year": d.year}, date_list)
    reduced = reduce(lambda acc, val: acc + [val], mapped, [])
    return pd.DataFrame(reduced)

# 3) 
# Create a function called "compute_distance" that takes
# a list of tuple pairs with latitude and longitude coordinates and 
# returns a list with the distance between the two pairs
# example input: [((41.23,23.5), (41.5, 23.4)), ((52.38, 20.1),(52.3, 17.8))]
# HINT: You can use geopy.distance in order to compute the distance
from geopy.distance import distance
from functools import reduce

def compute_distance(coords):
    distances = map(lambda pair: distance(pair[0], pair[1]).km, coords)
    return list(distances)

#example
pairs = [((41.23, 23.5), (41.5, 23.4)), ((52.38, 20.1), (52.3, 17.8))]
print(compute_distance(pairs))

#################################################
# 4)
# Consider a list that each element can be an integer or
# a list that contains integers or more lists with integers
# example: [[2], 4, 5, [1, [2], [3, 5, [7,8]], 10], 1]. 
# create a recursive function called "sum_general_int_list"
# that takes as input this type of list 
# and returns the sum of all the integers within the lists
# for instance for list_1=[[2], 3, [[1,2],5]] 
# the result should be 13

from functools import reduce

def sum_general_int_list(lst):
    return reduce(
        lambda acc, x: acc + (sum_general_int_list(x) if isinstance(x, list) else x),
        lst,
        0
    )

#example
list_1 = [[2], 3, [[1, 2], 5]]
print(sum_general_int_list(list_1))  

list_2 = [[2], 4, 5, [1, [2], [3, 5, [7, 8]], 10], 1]
print(sum_general_int_list(list_2))  


#data analysis
import pandas as pd
import diablib.data_prep as dp
import diablib.modeling as md
import diablib.evaluation as ev


df = dp.load_data("sample_diabetes_mellitus_data.csv")
df = dp.clean_data(df)
df = dp.encode_data(df)
train, test = dp.split_data(df)

model = md.train_model(train)
train = md.predict(model, train)
test = md.predict(model, test)

train_auc, test_auc = ev.evaluate_model(model, train, test, md.FEATURES, md.TARGET)
print(f"Train ROC-AUC: {train_auc:.3f}")
print(f"Test ROC-AUC: {test_auc:.3f}")






