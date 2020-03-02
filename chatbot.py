# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import movielens

import numpy as np
import re
import math
import collections
from PorterStemmer import PorterStemmer

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`. Give your chatbot a new name.
        self.name = 'moviebot'

        self.SPELL_CHECK_FLAG = False
        self.creative = creative

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = movielens.ratings()
        self.sentiment = movielens.sentiment()

        # Construct a stemmed sentiment lexicon
        self.stemmedsentiment = {}
        for word in self.sentiment:
            self.stemmedsentiment[PorterStemmer().stem(word)] = self.sentiment[word]

        #############################################################################
        # TODO: Binarize the movie ratings matrix.                                  #
        #############################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)

        self.user_ratings = np.zeros(len(self.ratings))
        self.rated_indices = []
        self.movies_processed = set([])
        self.already_recommended = set([])
        self.recommendations = collections.deque([])
        self.ASKED_FOR_REC = False
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    #############################################################################
    # 1. WARM UP REPL                                                           #
    #############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        #############################################################################
        # TODO: Write a short greeting message                                      #
        #############################################################################

        greeting_message = "Hi I'm MovieBot! I'm going to recommend a movie to you."

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return greeting_message

    def goodbye(self):
        """Return a message that the chatbot uses to bid farewell to the user."""
        #############################################################################
        # TODO: Write a short farewell message                                      #
        #############################################################################

        goodbye_message = "Have a nice day!"

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return goodbye_message

    ###############################################################################
    # 2. Modules 2 and 3: extraction and transformation                           #
    ###############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        #############################################################################
        # TODO: Implement the extraction and transformation in this method,         #
        # possibly calling other functions. Although modular code is not graded,    #
        # it is highly recommended.                                                 #
        #############################################################################
        # yes_phrases = set(['yes', 'yea', 'yeah', 'sure', 'yup', 'ok', 'okay'])
        yes_re = '.*(yes|yea|yeah|sure|yup|ok).*' 
        no_re = '.*(no|nah|negative).*'
        # no_phrases = set(['no', 'nope', 'nah', 'negative'])
        response = "baseic"

        if self.creative:       # CREATIVE MODE
            if self.SPELL_CHECK_FLAG:
                if line == 'yes':
                    response = 'Great, you like "The Notebook".'
                else:
                    response = "What did you like?"
                self.SPELL_CHECK_FLAG = False
            movie_titles = self.extract_titles(line)
            # find movies by title
            if len(movie_titles) == 1:
                if self.find_movies_by_title(movie_titles[0]) == []:
                    close_spellings = self.find_movies_closest_to_title(movie_titles[0])
                    print (close_spellings)
                    if len(close_spellings) > 0:
                        response = "Did you mean " + ' '.join(np.array(self.titles)[close_spellings, 0]) + ". Answer 'yes' or 'no'!"
                        self.SPELL_CHECK_FLAG = True
                    else:
                        response = "Sorry I don't know that movie"
            # response = "I processed {} in starter mode!!".format(line)
        else:                   # STARTER MODE
            if self.ASKED_FOR_REC:  # Expecting some variation of 'yes' or 'no' as an answer
                if re.match(yes_re, line.lower()):
                    return self.giveRecommendation()
                elif re.match(no_re, line.lower()):
                    self.ASKED_FOR_REC = False
                    return "Ok. Let's talk about more movies!"
                else:
                    return "I'm sorry, but I didn't quite understand your answer to my question. Would you like more recommendations--yes or no?"

            movie_titles = self.extract_titles(line)

            if len(movie_titles) > 1:        # Extracted multiple candidate movie titles from the line (basic mode doesn't handle this)
                return "Please tell me about one movie at a time. Go ahead."

            elif len(movie_titles) == 0:     # Extracted no candidate movie titles at all from the line
                # If the user's input text has the word recommend, assume they are asking for a recommendation 
                if line.lower().find('recommend') != -1 and len(self.movies_processed) < 5:     
                    return "Before I make any recommendations, I need to learn more about your preferences. Please tell me about another movie you liked."
                
                elif line.lower().find('recommend') != -1 and len(self.movies_processed) >= 5:
                    return self.giveRecommendation()

                else: # Otherwise, our chatbot is unable to process the user's message.
                    return "Sorry, I don't understand. Tell me about a movie that you have seen."

            else:                            # Extracted exactly one candidate movie title from the line
                title = movie_titles[0]
                movie_indices = self.find_movies_by_title(movie_titles[0])
                response = self.generateResponseStarter(title, movie_indices, line)

                # IF RESPONSE IS SOMEHOW INVALID, RETURN SOME GENERAL RESPONSE

                return response

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return response


    def generateResponseStarter(self, title, movie_indices, line):
        """Generate an appropriate chatbot response given a title, list of movie indices, and an input line

        According to spec for the starter implementation, if an input line yields 0 or >1 movie index, we are
        unable to process the information and return a response saying so.

        If there is exactly one movie index in movie_indices, we must process the sentiment. If the sentiment is
        non-neutral, we add it appropriately to self.user_ratings, and return a message which confirms that we 
        have processed the sentiment.

        Once we have processed >= 5 data points, the chatbot begins to offer recommendations to the user using the
        self.recommend function.
        """
        response = "Please tell me more about your movie preferences" # placeholder

        if len(movie_indices) == 0:         # If no valid movies were found from the title
            return "Sorry, I've never heard of a movie called \"{}\". Please tell me about another movie you liked.".format(title)

        elif len(movie_indices) > 1:        # If multiple valid movies were fround from the title
            return "I found more than one movie called \"{}\". Can you please clarify?".format(title)

        else:                               # Exactly one valid movie was found from the title
            movie_index = movie_indices[0]
            sentiment = self.extract_sentiment(line)

            if sentiment == 0:              # Neutral sentiment
                return "I'm sorry, I'm not sure if you liked \"{}\". Tell me more about it.".format(title)
            else:
                self.user_ratings[movie_index] = sentiment
                self.rated_indices.append(movie_index)
                self.movies_processed.add(movie_index)
                if len(self.movies_processed) >= 5:
                    self.recommendations = collections.deque(self.recommend(self.user_ratings, self.ratings))
                    if sentiment > 0:
                        return("Got it, you liked \"{}\"! Let me think...".format(title) + self.giveRecommendation())
                    else:   # sentiment < 0
                        return("I see, you didn't liked \"{}\". Let me think...".format(title) + self.giveRecommendation())
                if sentiment > 0:
                    return "OK, you liked \"{}\"! Tell me what you thought of another movie.".format(title)
                else:   # sentiment < 0
                    return "OK, so you didn't like \"{}\"! Tell me what you thought of another movie.".format(title)

    def giveRecommendation(self):
        """ Returns a message giving a single recommendation based on the data points already received

        This only recommends movies which have not previously been recommended. If the current list of recommended movies is 
        exhausted, we grab (a default of) the next 10 best recommendations and store that in self.recommendations
        """
        if len(self.recommendations) == 0:
            self.recommendations = collections.deque(self.recommend(self.user_ratings, self.ratings))
        next_recommendation = self.recommendations.popleft()
        # while next_recommendation in self.already_recommended:
        #     next_recommendation = self.recommendations.popleft()
        #     if len(self.recommendations) == 0:
        #         self.recommendations = collections.deque(self.recommend(self.user_ratings, self.ratings))
        self.already_recommended.add(next_recommendation)
        self.ASKED_FOR_REC = True
        return "OK, given what you have told me, I think that you might like \"{}\". Would you like another recommendation?".format(self.titles[next_recommendation][0])
            


    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information from a line of text.

        Given an input line of text, this method should do any general pre-processing and return the
        pre-processed string. The outputs of this method will be used as inputs (instead of the original
        raw text) for the extract_titles, extract_sentiment, and extract_sentiment_for_movies methods.

        Note that this method is intentially made static, as you shouldn't need to use any
        attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        #############################################################################
        # TODO: Preprocess the text into a desired format.                          #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to your    #
        # implementation to do any generic preprocessing, feel free to leave this   #
        # method unmodified.                                                        #
        #############################################################################

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess('I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        quotedFormat = '\"(.*?)\"'
        potential_titles = re.findall(quotedFormat, preprocessed_input)
        return potential_titles


    def prune_article(self, title):
        articles = ['a', 'an', 'the', 'la', 'le', "l'", 'les']
        #print(title.split())
        words = title.split()
        new_title = title
        if words[0].lower() in articles:
            #print("hiii")
            article = words[0]
            #print(article)

            new_title = title[len(article) + 1:] + ', ' + article
            # title = title[len(article) + 1:]
            # print(title)
        return new_title

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 1953]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        if self.creative:
            ids = []
            title = self.prune_article(title)
            for id, t in enumerate(self.titles):
                # ignore capitalization
                if t[0].lower().find(title.lower()) != -1:
                    next_char = t[0].lower().find(title.lower()) + len(title)
                    # disambiguate!
                    if next_char == len(t[0]) or not t[0][next_char].isalpha():
                        ids.append(id)

            # print (np.array(self.titles)[ids])
            return ids
            
        # Regex strings
        titleFormat = '(An|A|The)? ?(.+)'
        yearFormat = '(\(\d\d\d\d\))'
        yearFormatBeg = '^(\(\d\d\d\d\))'

        # Parse using RegEx
        titleParts = re.findall(titleFormat, title)

        # Will contain 3 parts: [particle, title, year]
        partsList = list(titleParts[0])
        year = ''
        # If the year is present in the movie title, parse it out of second part
        if re.match(yearFormat, partsList[1][-6:]):
            year = partsList[1][-6:]
            partsList[1] = partsList[1][:-7]
        # Add either empty string or the year to the third index of partsList
        partsList.append(year)

        formerTitle = title    # Necessary in case the particle isn't pushed to the end of title

        # Construct newly formatted title by various cases
        if partsList[0] and partsList[2]:
            title = '%s, %s %s' % (partsList[1], partsList[0], partsList[2])
        elif partsList[0] and not partsList[2]:
            title = '%s, %s' % (partsList[1], partsList[0])
        elif partsList[1] and not partsList[0] and not partsList[2]:
            title = partsList[1]

        ids = []
        for id, t in enumerate(self.titles):
            # Only append ids where our titleString is at the front, and directly followed by year (if applicable)

            # Checking re-formatted title
            if t[0].find(title) == 0:
                possibleYear = t[0][len(title) + 1:]
                if not possibleYear or re.match(yearFormatBeg, possibleYear):
                    ids.append(id)
                    continue
            # Checking regular title (in case the article doesn't get shifted to end)
            if t[0].find(formerTitle) == 0:
                possibleYear = t[0][len(formerTitle) + 1:]
                if not possibleYear or re.match(yearFormatBeg, possibleYear):
                    ids.append(id)
                    continue
        return ids


    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the text
        is super negative and +2 if the sentiment of the text is super positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess('I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        negation_words = ['not', 'never', 'nothing']

        words = re.sub("\".*?\"", "", preprocessed_input)   # Remove the singular movie title
        words = re.sub(" +", " ", words)                    # Remove extraneous spaces
        words = re.sub('[,.!?\\-]', "", words)              # Remove punctuation
        words = words.split()                               # Split the words into a list
        words = [word.lower() for word in words]            # Send all words to lowercase

        sentiment = 0       # Initial sentiment (will toggle up and down)
        invert_flag = 1     # Whether to flip sentiment due to the presence of negation words

        for i, word in enumerate(words):
            # Invert the invert flag if you encounter a negation word
            if i > 0 and words[i] in negation_words or words[i].endswith('\'t'):
                invert_flag *= -1
                continue

            # Check if the word itself is in self.sentiment
            if word in self.sentiment:
                if self.sentiment[word] == 'pos':
                    sentiment += 1 * invert_flag
                else:       # self.sentiment[word] == 'neg'
                    sentiment -= 1 * invert_flag
                if invert_flag == -1:
                    invert_flag = 1
                continue

            # Check if the word itself is in self.stemmedsentiment
            if word in self.stemmedsentiment:
                if self.stemmedsentiment[word] == 'pos':
                    sentiment += 1 * invert_flag
                else:       # self.sentiment[word] == 'neg'
                    sentiment -= 1 * invert_flag
                if invert_flag == -1:
                    invert_flag = 1
                continue

            # Check if the stemmed word is in self.sentiment
            stemmed_word = PorterStemmer().stem(word)
            if stemmed_word in self.sentiment:
                if self.sentiment[stemmed_word] == 'pos':
                    sentiment += 1 * invert_flag
                else:       # self.sentiment[word] == 'neg'
                    sentiment -= 1 * invert_flag
                if invert_flag == -1:
                    invert_flag = 1
                continue

            # Check if the stemmed word is in self.stemmedsentiment
            if stemmed_word in self.stemmedsentiment:
                if self.stemmedsentiment[stemmed_word] == 'pos':
                    sentiment += 1 * invert_flag
                else:       # self.sentiment[word] == 'neg'
                    sentiment -= 1 * invert_flag
                if invert_flag == -1:
                    invert_flag = 1
                continue

        # Return the sentiment (1 for positive, -1 for negative, 0 for neutral)
        if sentiment > 0:
            return 1
        elif sentiment < 0:
            return -1
        else:   # sentiment == 0
            return 0


    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of pre-processed text
        that may contain multiple movies. Note that the sentiments toward
        the movies may be different.

        You should use the same sentiment values as extract_sentiment, described above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess('I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie title,
          and the second is the sentiment in the text toward that movie
        """
        return []

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least edit distance
        from the provided title, and with edit distance at most max_distance.

        - If no movies have titles within max_distance of the provided title, return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given title
          than all other movies, return a 1-element list containing its index.
        - If there is a tie for closest movie, return a list with the indices of all movies
          tying for minimum edit distance to the given movie.

        Example:
          chatbot.find_movies_closest_to_title("Sleeping Beaty") # should return [1656]

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title and within edit distance max_distance
        """
        def compute_edit_distance(a, b, max_dist):
            D = np.zeros((len(a)  + 1, len(b)  + 1))
            for i in range(len(a) + 1):
                D[i,0] = i
            for j in range(len(b) + 1):
                D[0,j] = j

            exit = False
            for i in range(1, len(a) + 1):
                for j in range(1, len(b) + 1):
                    step = 0 # if same char
                    if a[i - 1] != b[j - 1]:
                        step = 2
                    D[i,j] = min(D[i-1,j] + 1, D[i, j-1] + 1, D[i-1, j-1] + step)
                    
            return D[i,j]

        title = self.prune_article(title)
        cands = []
        for i, t in enumerate(self.titles):
            # take only the "normal spelling". drop the year and alt name
            if t[0][0] != '(':
                correct_spelling = t[0].split('(')[0][:-1]
            # if leading '(', take the first two 
            else:
                correct_spelling = '('.join(t[0].split('(')[0:2])[:-1]

            dist = compute_edit_distance(correct_spelling.lower(), title.lower(), max_distance)
            if dist <= max_distance:
                cands.append([i, dist])
        
        # no movies within min dist.
        if cands == []:
            return []

        sorted_by_dist = sorted(cands, key=lambda x:x[1])
        # print ('hello world')
        # print (sorted_by_dist)
        min_dist = sorted_by_dist[0][1]

        closest = [m[0] for m in sorted_by_dist if m[1] == min_dist]
        return closest

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be talking about
        (represented as indices), and a string given by the user as clarification
        (eg. in response to your bot saying "Which movie did you mean: Titanic (1953)
        or Titanic (1997)?"), use the clarification to narrow down the list and return
        a smaller list of candidates (hopefully just 1!)

        - If the clarification uniquely identifies one of the movies, this should return a 1-element
        list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it should return a list
        with the indices it could be referring to (to continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by the clarification
        """
        ids = []
        data = []
        filler_words = ['the', 'one']
        result_clarification = ' '.join([w for w in clarification.split() if w not in filler_words])
        
        for cand in candidates:
            # grab only the movie title part. Drop the year and the final space space
            title = '('.join(self.titles[cand][0].split('(')[:-1])[:-1]
            # extract year and drop final ")"
            year  = self.titles[cand][0].split('(')[-1][:-1]
            if title.find(clarification) != -1 or year == clarification or title.find(result_clarification) != -1:
                ids.append(cand)
            
            data.append([cand, year])
        # Outputs the second movie in the list. However, if the clarification is "2" for ["Scream 2 (1997)", "Scream 1 (1996)"], it should still output "Scream 2 (1997)" since "2" is a substring.
        if ids == [] and clarification.isnumeric():
            return [candidates[int(clarification) - 1]]

        sorted_by_year = sorted(data, key=lambda x:x[1])

        if clarification == 'most recent':
            return [sorted_by_year[-1][0]]

        # second one case
        order_words = {
            'first': 0,
            'second': 1,
            'third': 2,
            'fourth': 3,
            'fifth': 4,
            'sixth': 5,
            'seventh': 6,
            'eighth': 7,
            'ninth': 8,
        }

        for order_word in order_words.keys():
            if order_word in clarification:
                return [candidates[order_words[order_word]]]

        return ids

    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use any
        attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from 0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered positive

        :returns: a binarized version of the movie-rating matrix
        """
        #############################################################################
        # TODO: Binarize the supplied ratings matrix. Do not use the self.ratings   #
        # matrix directly in this function.                                         #
        #############################################################################

        # The starter code returns a new matrix shaped like ratings but full of zeros.
        binarized_ratings = np.zeros_like(ratings)
        for row in range(len(ratings)):
            for col in range(len(ratings[0])):
                if ratings[row][col] == 0:
                    continue
                if ratings[row][col] > threshold:
                    binarized_ratings[row][col] = 1
                else:
                    binarized_ratings[row][col] = -1

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        #############################################################################
        # TODO: Compute cosine similarity between the two vectors.
        #############################################################################
        similarity = 0
        denominator = np.linalg.norm(u) * np.linalg.norm(v)
        if denominator == 0:
            return 0 
        numerator = np.dot(u, v)
        similarity = numerator / denominator
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in ratings_matrix,
          in descending order of recommendation
        """

        #######################################################################################
        # TODO: Implement a recommendation function that takes a vector user_ratings          #
        # and matrix ratings_matrix and outputs a list of movies recommended by the chatbot.  #
        # Do not use the self.ratings matrix directly in this function.                       #
        #                                                                                     #
        # For starter mode, you should use item-item collaborative filtering                  #
        # with cosine similarity, no mean-centering, and no normalization of scores.          #
        #######################################################################################

        # Create a dense array containing movie ratings from user, keeping track of their indices.
        rated_movies = []
        dense_user_ratings = []
        print('ok')

        for i, rating in enumerate(user_ratings):
            if rating != 0:
                rated_movies.append(i)
                dense_user_ratings.append(rating)

        # Predict ratings based on movie similarities
        recommendations = []
        ratingsList = []
        for i in range(len(user_ratings)):
            # If user has already rated this movie or we have already recommended this movie, continue
            if user_ratings[i] != 0 or i in self.already_recommended:
                continue
            rating = np.dot(np.array([self.similarity(ratings_matrix[i], ratings_matrix[j]) for j in rated_movies]), dense_user_ratings)
            ratingsList.append((rating, i))
            
        ratingsList.sort(reverse=True)
        recommendations = [pair[1] for pair in ratingsList[:k]]      
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return recommendations

    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, line):
        """Return debug information as a string for the line string from the REPL"""
        # Pass the debug information that you may think is important for your
        # evaluators
        debug_info = 'debug info'
        return debug_info

    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your chatbot
        can do and how the user can interact with it.
        """
        return """
        Your task is to implement the chatbot as detailed in the PA6 instructions.
        Remember: in the starter mode, movie names will come in quotation marks and
        expressions of sentiment will be simple!
        Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, run:')
    print('    python3 repl.py')
