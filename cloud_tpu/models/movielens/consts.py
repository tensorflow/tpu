# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines constants used in the movielens model sample code."""

from tensorflow.contrib.learn.python.learn.estimators.model_fn import ModeKeys

TRAIN, EVAL, INFER = ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.INFER

MODEL_TYPES = ['matrix_factorization', 'dnn_softmax']
MATRIX_FACTORIZATION, DNN_SOFTMAX = MODEL_TYPES

EVAL_TYPES = ['regression', 'ranking']
REGRESSION, RANKING = EVAL_TYPES

EMBEDDING_WEIGHT_INITIALIZERS = ['truncated_normal']
TRUNCATED_NORMAL, = EMBEDDING_WEIGHT_INITIALIZERS

OPTIMIZERS = ['Adagrad', 'Adam', 'RMSProp']
ADAGRAD, ADAM, RMSPROP = OPTIMIZERS

MOVIE_VOCAB_SIZE = 28000
GENRE_VOCAB_SIZE = 20
# Use for initializing the global_rating_bias variable.
RATING_BIAS = 3.5

# Define a output alternative key for generating the top 100 candidates
# in exported model.
DEFAULT_OUTPUT_ALTERNATIVE = 'candidate_gen_100'

# Repeat the constants defined in movielens.py here as task.py will not be able
# to access movielens.py in SDK integration test.
"""Names of feature columns associated with the `Query`. These are the features
typically included in a recommendation request. In the case of movielens,
query contains just data about the user. In other applications, there
could be additional dimensions such as context (i.e. device, time of day, etc).
"""
# The user id.
QUERY_USER_ID = 'query_user_id'
# The ids of movies rated by the user.
QUERY_RATED_MOVIE_IDS = 'query_rated_movie_ids'
# The scores on the rated movies given by the user.
QUERY_RATED_MOVIE_SCORES = 'query_rated_movie_scores'
# The set of genres of the rated movies.
QUERY_RATED_GENRE_IDS = 'query_rated_genre_ids'
# The number of times the user rated each genre.
QUERY_RATED_GENRE_FREQS = 'query_rated_genre_freqs'
# The average rating on each genre.
QUERY_RATED_GENRE_AVG_SCORES = 'query_rated_genre_avg_scores'


"""Names of feature columns associated with the `Candidate`. These features
are used to match a candidate against the query."""
# The id of the candidate movie.
CANDIDATE_MOVIE_ID = 'cand_movie_id'
# The set of genres of the candidate movie.
CANDIDATE_GENRE_IDS = 'cand_genre_ids'
# The ranking candidate movie ids used to rank candidate movie against (used
# only in Eval graph).
RANKING_CANDIDATE_MOVIE_IDS = 'ranking_candidate_movie_ids'


"""Names of feature columns defining the label(s), which indicates how well
a candidate matches a query. There could be multiple labels in each instance.
Eg. We could have one label for the rating score and another label for the
number of times a user has watched the movie."""
LABEL_RATING_SCORE = 'label_rating_score'
