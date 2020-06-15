import numpy
import pandas
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

#features of input file
#id,betweenness,closeness,follow,posts,interval,length,movie_posts,reposts,coapper_posts,boxoffice,fans,verifyName
data = pandas.read_csv('actor-network-characteristics.csv', sep=',', dtype={'verifyName':object}, encoding='gbk')
#actor fans in 10K, this file is target category.
target = pandas.read_csv('actor_influence_target.csv', header=None, sep=',', names=['fans'])

try:
    matrix = joblib.load("extra.npz")
except:
    print('First only.')
    # filter.csv is used for align network features with all pairs shortest path matrx.
    np = pandas.read_csv("./filter.csv", sep=',', header=None)
    np[:] = np[:] - 2
    shortestpath = pandas.read_csv('./memory.txt', sep='  ', header=None) # memory.txt is too large to upload. use 'extra.npz' for all the time.
    new_index = pandas.Int64Index(numpy.arange(len(shortestpath))).difference(np[0].values.tolist())
    filtered = shortestpath.loc[new_index, new_index]
    joblib.dump(filtered.values, "extra.npz", compress=3)
    matrix = filtered.values

extramatrix = pandas.DataFrame(matrix)

dense_features = ['betweenness', 'closeness', 'follow', 'posts', 'interval', 'length', 'movie_posts', 'reposts', 'coapper_posts', 'boxoffice']
print(dense_features)
fulldata = pandas.concat([data[dense_features], extramatrix], axis=1)

X_train, X_test, y_train, y_test =  train_test_split(fulldata, target, test_size=0.2, random_state=1234)

#if training a model on training set, change to X_train and y_train, change this line. Now use the full dataset for training.
gbt = GradientBoostingClassifier(learning_rate=0.01, n_estimators=100, max_depth=9, min_samples_leaf =80,
                                 min_samples_split =1000,
                                 max_features=9, verbose=1,
                                 # subsample=0.7
                                 # random_state=10
                                 ).fit(fulldata, target)

joblib.dump(gbt, 'gbt.model')

fullp = gbt.predict(fulldata)
print("training score : %.3f " % gbt.score(fulldata, target))
print("Mean squared error: %.2f" % mean_squared_error(fullp, target))
print('Variance score: %.2f' % r2_score(fullp, target))

fullpdf = pandas.DataFrame(fullp, columns=['influence'])
output = pandas.concat([fullpdf, data['verifyName'], target], axis=1)
output = pandas.DataFrame(output, columns=['influence','verifyName', 'fans'])
output.to_csv('output-influence-with-name-target.csv', encoding='gbk', columns=['influence','verifyName', 'fans'], header=True, index=False)
print('INF generated.')