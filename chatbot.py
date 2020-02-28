# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import movielens

import numpy as np
import re
import math
from PorterStemmer import PorterStemmer


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`. Give your chatbot a new name.
        self.name = 'moviebot'

        self.creative = creative

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = movielens.ratings()
        self.sentiment = movielens.sentiment()

        #############################################################################
        # TODO: Binarize the movie ratings matrix.                                  #
        #############################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = ratings
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
        if self.creative:
            response = "I processed {} in starter mode!!".format(line)
        else:
            movie_titles= self.extract_titles(line)
            if len(movie_titles) > 1:
                return "Please tell me about one movie at a time. Go ahead."
            elif len(movie_titles) == 0:
                return "Sorry, I don't understand. Tell me about a movie that you have seen."
            else:
                self.find_movies_by_title(movie_titles[0])
                self.extract_sentiment(line)


                response = "I processed {} in creative mode!!".format(line)

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return response

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
        print(potential_titles)
        return potential_titles

        #return []

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
<<<<<<< HEAD
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
        print(partsList)

        formerTitle = title    # Used in case the particle isn't pushed to the end of title

        # Construct newly formatted title by various cases
        if partsList[0] and partsList[2]:
            title = '%s, %s %s' % (partsList[1], partsList[0], partsList[2])
        elif partsList[0] and not partsList[2]:
            title = '%s, %s' % (partsList[1], partsList[0])
        elif partsList[1] and not partsList[0] and not partsList[2]:
            title = partsList[1]

        ids = set([])
        for id, t in enumerate(self.titles):
            # Only append ids where our titleString is at the front, and directly followed by year (if applicable)
            if t[0].find(title) == 0:
                possibleYear = t[0][len(title) + 1:]
                if not possibleYear or re.match(yearFormatBeg, possibleYear):
                    ids.add(id)
            if t[0].find(formerTitle) == 0:
                possibleYear = t[0][len(formerTitle) + 1:]
                if not possibleYear or re.match(yearFormatBeg, possibleYear):
                    ids.add(id)
            # if t[0].find(title) == 0 or t[0].find(formerTitle) == 0:
            #     ids.append(id)
        return list(ids)

        '''
        moviesDatabase = open('data/movies.txt', 'r')
        words = title.split()
        if words[0] in ['a', 'an', 'the']:
            title = word[1:] + word[0]
            print(title)
        print(title)
        for line in moviesDatabase:
            startIndex = line.find('%') + 1
            #yearInQuotes = line.find(r'(\d\d\d\d)', line)
            #print(yearInQuotes)
            #endIndex = 
        return []
        '''

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
        #print(self.sentiment)
        #print(self.sentiment['unspeakable'])
        print(preprocessed_input)
        words = re.sub("\".*?\"", "", preprocessed_input)
        words = words.split()
        words = preprocessed_input.split()
        fillerWords = ['really', 'actually', 'very', 'honestly', 'extremely', 'that']

        #remove filler words from list of words
        words = [word for word in words if word not in fillerWords]
        #print(words)
        stemmedLexicon = {}
        for word in self.sentiment:
            stemmedLexicon[PorterStemmer().stem(word)] = self.sentiment[word]

        #print(stemmedLexicon)
        #print(words)
        posWordCount = 0
        negWordCount = 0
        for i, word in enumerate(words):
            wordStem = PorterStemmer().stem(word)
            #print(wordStem)
            if wordStem in stemmedLexicon:
                if i > 0 and (words[i-1] in ['not', 'never', 'nothing'] or words[i-1].endswith('n\'t')):
                    # if i > 1 and (words[i-2] in ['not', 'never', 'nothing'] or words[i-2].endswith('n\'t')):
                    #     if stemmedLexicon[wordStem] == 'pos':
                    #         posWordCount += 1
                    #     else:
                    #         negWordCount += 1
                    #     continue
                    if stemmedLexicon[wordStem] == 'pos':
                        negWordCount += 1
                    else:
                        posWordCount += 1
                else:
                    if stemmedLexicon[wordStem] == 'pos':
                        posWordCount += 1
                    else:
                        negWordCount += 1
        if posWordCount > negWordCount:
            print("positive")
            return 1
        if posWordCount < negWordCount:
            print("negative")
            return -1
        if posWordCount == negWordCount:
            print("neutral")
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
        pass

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

        pass

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
        pass

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
                if ratings[row][col] != 0 and ratings[row][col] <= threshold:
                    binarized_ratings[row][col] = -1
                elif ratings[row][col] > threshold:
                    binarized_ratings[row][col] = 1
        #print(binarized_ratings)

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
        #print(u)
        #print(v)
        similarity = 0
        numerator = np.dot(u, v)
        denominator = math.sqrt(np.dot(u, u)) * math.sqrt(np.dot(v,v)) 
        similarity = numerator / denominator
        #print(similarity)
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

        # Populate this list with k movie indices to recommend to the user.
        recommendations = []
        ratingsList = []
<<<<<<< HEAD
        # print(user_ratings)
        # print(ratings_matrix)
        # print(len(ratings_matrix))
        for i in range(len(ratings_matrix)):
            rating = np.dot(np.array([self.similarity(ratings_matrix[i], ratings_matrix[j]) if movieRating != 0 else 0 for j, movieRating in enumerate(user_ratings)]), user_ratings)
            # for j, movieRating in enumerate(user_ratings):
            #     if movieRating != 0:
            #         similarity = self.similarity(ratings_matrix[i], ratings_matrix[j])
            #         rating += similarity * movieRating

=======
        #print(user_ratings)
        #print(ratings_matrix)
        for i in range(len(ratings_matrix)):
            rating = 0
            for j, movieRating in enumerate(user_ratings):
                if movieRating != 0:
                    similarity = self.similarity(ratings_matrix[i], ratings_matrix[j])
                    rating += similarity * movieRating
>>>>>>> c001da67aef322488caed3e1496215db3ea8e5a8
            #print(rating)
            #only want to append ratings on movies that the user hasn't already seen
            if user_ratings[i] == 0:
                ratingsList.append((rating, i))
<<<<<<< HEAD
        # print(ratingsList)
        ratingsList.sort(reverse=True)
        # print(ratingsList)
        # if len(ratingsList) >= k:
        recommendations = [pair[1] for pair in ratingsList[:k]]            
=======
        ratingsList.sort(reverse=True)
        #print(ratingsList)
        if len(ratingsList) >= k:
            recommendations = [pair[1] for pair in ratingsList[:k]]
>>>>>>> c001da67aef322488caed3e1496215db3ea8e5a8

        print(recommendations)
         
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
