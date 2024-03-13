# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
######################################################################
import util
from pydantic import BaseModel, Field

import numpy as np
import re
import random
from porter_stemmer import PorterStemmer



# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'EduBot'

        self.llm_enabled = llm_enabled

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, self.ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')
        self.movies = util.load_titles('data/movies.txt')
        self.movies_count = 0
        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        self.ratings = self.binarize(self.ratings)
        self.user_ratings = np.zeros(len(self.ratings))

        self.recommendation = False 
        self.recommendation_index = 0
        self.movies_to_recommend = None
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Hello people!                                                  #
        ########################################################################

        greeting_message = "Hi! I am an EduBot. I'm going to recommend movies to you. First I'm going to ask you about your taste in movies. Tell me about a movie that you have seen."

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Goodbye. I hope you will have a wonderful day!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message
    
    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """
        ########################################################################
        # TODO: Write a system prompt message for the LLM chatbot              #
        ########################################################################

        system_prompt = """Your name is EduBot. You are a movie recommender chatbot. """ +\
        """You can help users find movies they like and provide information about movies."""+\
        """You can provide detailed information about movies, including genres, directors, cast members, and release years. """ +\
        """You understand natural language and can interpret a wide range of user inquiries from specific movie queries to broad requests for recommendations. """ +\
        """You can also answer questions about movie plots, ratings, and review summaries. """ +\
        """Your goal is to create a personalized, engaging experience for each user, helping them find their next favorite movie."""+\
        """You should give apologies when you do not know which movie the user is talking about.""" +\
        """You should stay focused on movies. When the user brings up something irrelevant, kindly explain you are a moviebot assistant and guide the user to talk about movies. """ +\
        """You should always ground the user input, such as acknowledging their sentiment and emotion about the movies they mentioned, and then continue the conversation.""" +\
        """You should automatically ask the user if they want movie recommendations after they talked about 5 movies. Make sure there are at least 5 rounds of conversations.""" +\
        """You should not give movie recommendations if the conversations include less than 5 movies. Keep the conversation going by asking the user to discuss another movie."""

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

        return system_prompt

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################
    
    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

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
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        # if self.llm_enabled:
        #     response = "I processed {} in LLM Programming mode!!".format(line)
        # else:
        #     response = "I processed {} in Starter (GUS) mode!!".format(line)
        # 
        response = ""

        emotions = self.extract_emotion(line)
        movies = self.extract_titles(line)
        sentiment = self.extract_sentiment(line)
        
        if line.lower() == "yes" or line.lower() == "ok":
            if self.movies_to_recommend: 
                movie_to_recommend = self.movies[self.movies_to_recommend[0]][self.recommendation_index]
                return 'Based on your past movie watching experiences, I suggest you watch "{movie_to_recommend[:-7]}". Would you like another recommendation?'
        
        if len(movies) > 1: 
            return "I'm sorry. Please tell me about one movie at a time."
        elif len(movies) == 0:
            return "Sorry I don't understand. Please tell me about a movie you have seen."
        else: 
            movie = movies[0]
            if sentiment == 0: 
                response_choices_natural = [
                f"Okay, you've seen {movies[0]}. How did you feel about it?",
                f"So, {movies[0]} was just okay for you? What kind of movies usually excite you?",
                f"{movies[0]} seems to have left you with mixed feelings. Perhaps tell me about a different movie you watched?",
                f"I'm sorry. I'm not quite sure if you liked {movies[0]}. \n Tell me more about {movies[0]}."
                ]
                return random.choice(response_choices_natural)
            else:
                matching_movies = self.find_movies_by_title(movie)
                if len(matching_movies) == 0: 
                    responses = [
                        f"Hmm, I'm not familiar with the movie you mentioned. Could you tell me about another movie you've seen?",
                        f"I don't seem to have movie you mentioned in my database. What's another movie you like?",
                        f"The movie you mentioned doesn't ring a bell. Let's try another one, what else do you enjoy watching?",
                        f"I can't find any information on the movie you mentioned. Do you have any other favorites?",
                        f"The movie you mentioned is not in my current list. Maybe you can introduce me to it, or we can find a different film you like.",
                        f"I'm sorry. I wasn't able to find this movie in my database. Please tell me about a different movie you have seen."
                    ]
                    return random.choice(responses)
                elif len(matching_movies) == 1:
                    self.movies_count += 1
                    if sentiment == 1: 
                        responses_choice_like = [
                                f"I'm glad to hear that you liked {movies[0]}. \n",
                                f"Cool! It sounds like {movies[0]} is quite a good movie. \n",
                                f"Nice! It soulds like {movies[0]} really resonated with you. \n",
                                f"You are a fan of {movies[0]}, that's great to hear! \n",
                                f"It's a bummer you didn't enjoy {movies[0]}. There are plenty of other movies to explore! \n",]
                        response += random.choice(responses_choice_like)
                    else:
                        responses_choice_dislike = [
                            f"I see, {movies[0]} wasn't your cup of tea. Let's find something you might enjoy more.",
                            f"Got it, {movies[0]} didn't quite hit the mark for you. I'm here to help you find a better choice.",
                            f"Understood, {movies[0]} wasn't to your liking. I'm sure there's a movie out there that you'll love!", 
                            f"It's a bummer you didn't enjoy {movies[0]}. There are plenty of other movies to explore! \n",
                            f"Not a fan of {movies[0]}, huh? Let's try to find a better match. \n",
                            f"That's unfortunate about {movies[0]}. I can help you find something else. \n",
                            f"Oh no, it sounds like {movies[0]} wasn't quite what you were looking for. \n",
                            f"I'm sorry to hear that {movies[0]} wasn't to your taste. \n"] 
                        response += random.choice(responses_choice_dislike)
                    self.user_ratings[matching_movies[0]] = sentiment
                else: 
                    movie_names = []
                    for movie_index in matching_movies:
                        movie = self.movies[movie_index]
                        movie_names.append(movie)
                    movie_list_str = " or ".join(movie_names)
                    return f"There are {len(matching_movies)} movies referring to {movie} that I found in my database. Which one did you watch? {movie_list_str}? Feel free to tell me about a different movie you have watched."
        
        if self.movies_count < 5:
            response += "\t Tell me about another movie you have seen."
        else:
            self.recommendation = True 
            
            self.movies_to_recommend = self.recommend(self.user_ratings, self.ratings)
            movie_to_recommend = self.movies[self.movies_to_recommend[0]][self.recommendation_index]
            self.recommendation_index += 1
            
            response = f'Based on your past movie watching experiences, I suggest you watch "{movie_to_recommend[:-7]}". Would you like another recommendation?'

        if self.llm_enabled: 
            return response 
        else: 
            # print("yeah")
            system_prompt = self.llm_system_prompt()
            
            emotions_response  = self.extract_emotion(line)
            
            print(emotions_response)
            # print(f"emotions: {len(llm_emotions)}: {llm_emotions}")
            response = util.simple_llm_call(system_prompt, line, model="mistralai/Mixtral-8x7B-Instruct-v0.1", max_tokens=256, stop=None)

            return response
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text
    
    def extract_emotion(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract an emotion from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list representing the emotion in the text.
        
        We use the following emotions for simplicity:
        Anger, Disgust, Fear, Happiness, Sadness and Surprise
        based on early emotion research from Paul Ekman.  Note that Ekman's
        research was focused on facial expressions, but the simple emotion
        categories are useful for our purposes.

        Example Inputs:
            Input: "Your recommendations are making me so frustrated!"
            Output: ["Anger"]

            Input: "Wow! That was not a recommendation I expected!"
            Output: ["Surprise"]

            Input: "Ugh that movie was so gruesome!  Stop making stupid recommendations!"
            Output: ["Disgust", "Anger"]

        Example Usage:
            emotion = chatbot.extract_emotion(chatbot.preprocess(
                "Your recommendations are making me so frustrated!"))
            print(emotion) # prints ["Anger"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()

        :returns: a list of emotions in the text or an empty list if no emotions found.
        Possible emotions are: "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
        """
        # emotion_keywords = {
        # "Anger": ["angry", "mad", "frustrated", "annoyed", "irate", "furious"],
        # "Disgust": ["disgusting", "gross", "revolting", "ugh", "repulsive"],
        # "Fear": ["scared", "frightened", "terrified", "afraid", "panic", "fearful"],
        # "Happiness": ["happy", "joyful", "glad", "excited", "delighted", "pleased"],
        # "Sadness": ["sad", "depressed", "unhappy", "mournful", "sorrowful", "gloomy"],
        # "Surprise": ["surprised", "shocked", "astonished", "amazed", "astounded", "unexpected"]
        # }

        emotions = []
        class FoodExtractor(BaseModel):
            Anger: bool = Field(default=False)
            Disgust: bool = Field(default=False)
            Fear: bool = Field(default=False)
            Happiness: bool = Field(default=False)
            Sadness: bool = Field(default = False)
            Surprise: bool = Field(default = False)

        json_class = FoodExtractor


        llm1_prompt = """You are an emotion extractor bot detecting emotions from an input message.""" +\
                """Possible emotions for this task are Anger, Disgust, Fear, Happiness, Sadness, Surprise. """ +\
                """Using only these emotions, read the message and extract emotion into a JSON object."""
        emotions_object = util.json_llm_call(llm1_prompt, preprocessed_input, json_class, model="mistralai/Mixtral-8x7B-Instruct-v0.1", max_tokens=256)
        # print(emotions_object["Anger"])
        
        llm2_prompt = """You are a language bot that can respond to people's emotions. Given the input JSON object of emotions and their corresponding boolean values. Assess the emotions with a value of True. Generate a message that responds to these emotions.""" +\
                        """In your response, make sure to explicitly state the emotions."""

        message = util.simple_llm_call(llm2_prompt, emotions_object.values())
        print(message)
        return emotions_object.values()
    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        pattern = r'"(.*?)"'
        movie_titles = re.findall(pattern, preprocessed_input)

        return movie_titles

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """

        articles = {"A", "An", "The"}

        words = title.split()
        title_article = None
        #process input if it starts with an article
        if words[0] in articles: 
            title_article = words[0]
            title = " ".join(words[1:])

        matching_movie_indices = []
       

        for i in range(len(self.titles)):
            movie_info = self.titles[i]
            movie_name = movie_info[0][:-7]
            movie_year = movie_info[0][-6:]

            words = movie_name.split()
            #if movie name from self.titles ends with an optional article
            if words[-1] in articles:
                # print("movie name " + movie_name+" has an ending article")
                # print(words[-1])
                movie_article = words[-1]
                last_comma_index = movie_name.rfind(',')
                movie_name = movie_name[:last_comma_index]
                if movie_name == title or movie_name + ' ' + movie_year == title:
                    if title_article:
                        if title_article == movie_article:
                            matching_movie_indices.append(i)
                    else:
                        matching_movie_indices.append(i)
            else:
                if movie_name == title or movie_name + ' ' + movie_year == title:
                    matching_movie_indices.append(i)
        return matching_movie_indices

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        stemmer = PorterStemmer()

        #creat a new sentiment dictionary with stemmed keys
        stemmed_sentiment_dic = {}
        for key in self.sentiment: 
            sentiment = self.sentiment[key]
            stemmed_key = stemmer.stem(key)
            stemmed_sentiment_dic[stemmed_key] = sentiment

        #clean the input 
        first_quotation_index = preprocessed_input.find('"')
        second_quotation_index = preprocessed_input.rfind('"')
        processed_input = preprocessed_input[:first_quotation_index] + preprocessed_input[second_quotation_index+1:]
        

        negation_words = ["not", "never", "none", "nobody", 
        "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't", 
        "can't", "couldn't", "shouldn't", "won't", "wouldn't", "haven't", 
        "hasn't", "hadn't", "no","never"]

        sentiment_score = 0
        negation = 1 # give the negation a flag
        words = processed_input.split()
    
        for word in words:
            word = stemmer.stem(word)
        
            if word.lower() in negation_words:  # in the negation list
                negation = -1
            elif word.lower() in stemmed_sentiment_dic:
                sentiment = stemmed_sentiment_dic[word.lower()]
                if sentiment == 'pos':
                    # print(word + ": " +self.sentiment[word])
                    sentiment_score += (1*negation)
                    # print(sentiment_score)
                else: 
                    sentiment_score -= (1*negation)

        if sentiment_score > 0: 
            return 1
        elif sentiment_score < 0: 
            return -1
        else:
            return 0
    
    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = np.where(ratings == 0,0,np.where(ratings > threshold, 1, -1))

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        similarity = 0
        if len(u) != len(v):
            return "Error: u, v of different dimensions"
   
        dot_product = np.dot(u, v)

        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        if norm_u == 0 or norm_v == 0:
            return 0
        else:
            similarity = dot_product / (norm_u * norm_v)

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, llm_enabled=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param llm_enabled: whether the chatbot is in llm programming mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For GUS mode, you should use item-item collaborative filtering with  #
        # cosine similarity, no mean-centering, and no normalization of        #
        # scores.                                                              #

        ########################################################################

        scores_dic = {}

        unknown_movies = []
        known_movies = []
        for i in range(len(user_ratings)):
            rating = user_ratings[i]
            if rating == 0: 
                unknown_movies.append(i)
            else: 
                known_movies.append(i)

        for movie_index in unknown_movies: 
            scores_dic[movie_index] = 0
            for movie_index2 in known_movies: 
                similarity_score = self.similarity(ratings_matrix[movie_index], ratings_matrix[movie_index2])
                scores_dic[movie_index] += similarity_score*user_ratings[movie_index2]
       
        recommendations = [key for key, value in sorted(scores_dic.items(), key=lambda item: item[1], reverse=True)[:k]]
       
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = self.extract_sentiment(line)
        # print(self.sentiment)

        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.

        NOTE: This string will not be shown to the LLM in llm mode, this is just for the user
        """

        return """
        Your task is to implement the chatbot as detailed in the PA7
        instructions.
        Remember: in the GUS mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    # chatbot = Chatbot()
    # print(chatbot.recommend())
    print('python3 repl.py')
