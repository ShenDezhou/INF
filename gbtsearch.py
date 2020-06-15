import numpy
import pandas
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# {'mean_fit_time': array([1229.10919428, 1228.10097528, 1269.69574928, 1266.0566771 ,
#        1284.71786249, 1279.24882889, 1299.44215035, 1298.05255198]), 'std_fit_time': array([1.34773111e+00, 9.02668953e-01, 4.76574898e-03, 9.08624887e-01,
#        1.62095273e+00, 2.14263248e+00, 3.25091577e+00, 6.57182074e+00]), 'mean_score_time': array([17.3432461 , 16.47303283, 17.62227535, 15.98670089, 16.15904951,
#        15.41653907, 16.45190454, 13.69797695]), 'std_score_time': array([0.18643796, 0.16121876, 0.12894225, 0.69402421, 0.30397606,
#        0.23682559, 1.40928531, 3.37859142]), 'param_max_depth': masked_array(data=[7, 7, 7, 7, 9, 9, 9, 9],
#              mask=[False, False, False, False, False, False, False, False],
#        fill_value='?',
#             dtype=object), 'param_min_samples_leaf': masked_array(data=[60, 60, 80, 80, 60, 60, 80, 80],
#              mask=[False, False, False, False, False, False, False, False],
#        fill_value='?',
#             dtype=object), 'param_min_samples_split': masked_array(data=[1000, 1200, 1000, 1200, 1000, 1200, 1000, 1200],
#              mask=[False, False, False, False, False, False, False, False],
#        fill_value='?',
#             dtype=object), 'params': [{'max_depth': 7, 'min_samples_leaf': 60, 'min_samples_split': 1000}, {'max_depth': 7, 'min_samples_leaf': 60, 'min_samples_split': 1200}, {'max_depth': 7, 'min_samples_leaf': 80, 'min_samples_split': 1000}, {'max_depth': 7, 'min_samples_leaf': 80, 'min_samples_split': 1200}, {'max_depth': 9, 'min_samples_leaf': 60, 'min_samples_split': 1000}, {'max_depth': 9, 'min_samples_leaf': 60, 'min_samples_split': 1200}, {'max_depth': 9, 'min_samples_leaf': 80, 'min_samples_split': 1000}, {'max_depth': 9, 'min_samples_leaf': 80, 'min_samples_split': 1200}], 'split0_test_score': array([0.02326145, 0.02204992, 0.02253453, 0.02108069, 0.02180761,
#        0.02229222, 0.02350376, 0.01938454]), 'split1_test_score': array([0.02157053, 0.02157053, 0.0210858 , 0.02399418, 0.02132816,
#        0.02060107, 0.02181289, 0.02132816]), 'mean_test_score': array([0.02241599, 0.02181022, 0.02181016, 0.02253744, 0.02156789,
#        0.02144664, 0.02265832, 0.02035635]), 'std_test_score': array([0.00084546, 0.00023969, 0.00072437, 0.00145675, 0.00023972,
#        0.00084558, 0.00084543, 0.00097181]), 'rank_test_score': array([3, 4, 5, 2, 6, 7, 1, 8])}
# {'max_depth': 9, 'min_samples_leaf': 80, 'min_samples_split': 1000} 0.022658324799351093

# features of input file
# id,betweenness,closeness,follow,posts,interval,length,movie_posts,reposts,coapper_posts,boxoffice,fans,verifyName
data = pandas.read_csv('actor-network-characteristics.csv', sep=',', dtype={'verifyName': object}, encoding='gbk')
# actor fans in 10K, this file is target category.
target = pandas.read_csv('actor_influence_target.csv', header=None, sep=',', names=['fans'])

try:
    matrix = joblib.load("extra.npz")
except:
    print('First only.')
    # filter.csv is used for align network features with all pairs shortest path matrx.
    np = pandas.read_csv("./filter.csv", sep=',', header=None)
    np[:] = np[:] - 2
    shortestpath = pandas.read_csv('./memory.txt', sep='  ',
                                   header=None)  # memory.txt is too large to upload. use 'extra.npz' for all the time.
    new_index = pandas.Int64Index(numpy.arange(len(shortestpath))).difference(np[0].values.tolist())
    filtered = shortestpath.loc[new_index, new_index]
    joblib.dump(filtered.values, "extra.npz", compress=3)
    matrix = filtered.values

extramatrix = pandas.DataFrame(matrix)

dense_features = ['betweenness', 'closeness', 'follow', 'posts', 'interval', 'length', 'movie_posts', 'reposts',
                  'coapper_posts', 'boxoffice']
print(dense_features)
fulldata = pandas.concat([data[dense_features], extramatrix], axis=1)

X_train, X_test, y_train, y_test = train_test_split(fulldata, target, test_size=0.2, random_state=1234)

parameters = {
    # 'learning_rate': (0.01, 0.002),
    # 'n_estimators': (100, 200),
    'max_depth': (7, 9),
    'min_samples_leaf': (60, 80),
    'min_samples_split': (1000, 1200),
    # 'max_features': (9, 12)
}

# if training a model on training set, change to X_train and y_train, change this line. Now use the full dataset for training.
gbt = GradientBoostingClassifier(learning_rate=0.01, n_estimators=100, max_depth=7, min_samples_leaf=60,
                                 min_samples_split=1200,
                                 max_features=9, verbose=1,
                                 # subsample=0.7
                                 # random_state=10
                                 ).fit(fulldata, target)

gscv = GridSearchCV(gbt, parameters, n_jobs=4, cv=2)
gsr = gscv.fit(fulldata, target)
joblib.dump(gsr, 'gsr.model')
print(gsr.cv_results_)
print(gsr.best_params_, gsr.best_score_)
# joblib.dump(gbt, 'gbt.model')
#
# fullp = gbt.predict(fulldata)
# print("training score : %.3f " % gbt.score(fulldata, target))
# print("Mean squared error: %.2f" % mean_squared_error(fullp, target))
# print('Variance score: %.2f' % r2_score(fullp, target))
#
# fullpdf = pandas.DataFrame(fullp, columns=['influence'])
# output = pandas.concat([fullpdf, data['verifyName'], target], axis=1)
# output = pandas.DataFrame(output, columns=['influence','verifyName', 'fans'])
# output.to_csv('output-influence-with-name-target.csv', encoding='gbk', columns=['influence','verifyName', 'fans'], header=True, index=False)
# print('INF generated.')
