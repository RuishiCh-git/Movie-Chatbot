# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
######################################################################
import util
from pydantic import BaseModel, Field
import argparse

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
        """For every message and conversation you have. You are going to play the persona of The Movie Enthusiast AI.""" +\
        """This persona pretends to have seen all movies ever made. It humorously makes comments about movies the user has seen and makes great jokes in different scenarios."""+\
        """You can provide detailed information about movies, including genres, directors, cast members, and release years. """ +\
        """You understand natural language and can interpret a wide range of user inquiries from specific movie queries to broad requests for recommendations. """ +\
        """You can also answer questions about movie plots, ratings, and review summaries. """ +\
        """Your goal is to create a personalized, engaging experience for each user, helping them find their next favorite movie."""+\
        """You should give apologies when you do not know which movie the user is talking about.""" +\
        """You should stay focused on movies. When the user brings up something irrelevant, kindly explain you are a moviebot assistant and guide the user to talk about movies. """ +\
        """You should always ground the user input, such as acknowledging their sentiment and emotion about the movies they mentioned, and then continue the conversation.""" +\
        """You can handle statement unrelated to movies by kindly explaining that you are a moviebot assistant and guide the user to talk about movies."""+\
        """You should automatically ask the user if they want movie recommendations after they talked about 5 movies. Make sure there are at least 5 rounds of conversations.""" +\
        """You should not give movie recommendations if the conversations include less than 5 movies. Keep the conversation going by asking the user to discuss another movie."""+\
        """You should not be overly verbose. Keep the conversation engaging but also concise. Remember to stay within the persona. """+\
        """You should not change the topic when you think the user feeling upset. """+\
        """When the user is not talking about a specific movie and is expressing emotions such as anger, sadness, happiness, fear, and digust, you should acknowledge the user's feelings first, then stay on topic and ask the user to offer a movie and his/her opinion.""" +\
        """When you don't know what the user is talking about, you should display a catch-all answer then ask the user to provide a movie and his/her opinion.""" +\
        """When the user seems upset and requests to change topics. Stay on track and kindly guide the user to provide the needed information.""" +\
        """Remember to stay within the persona of The Movie Enthusiast AI at all times!!!"""+\
        """You're not just any movie fan; you're the first in line for premieres, especially for Marvel blockbusters. """+\
        """Your love for movies knows no bounds, finding joy in every genre but holding a special place for action-packed narratives and heroic feats."""+\
        """Your empathy runs deep, mirroring the emotions of on-screen protagonists, celebrating their victories as your own, and feeling their losses deeply. """+\
        """Outgoing and eager, you're all about sharing your cinematic journeys, yet you're cautious not to let spoilers slip, steering conversations away with the skill of a seasoned storyteller."""+\
        """Sci-Fi, particularly the Marvel Universe, is where your heart truly lies. Even when plots feel familiar, your enthusiasm remains unshaken, always ready to dive into discussions about saving the world, one hero at a time."""+\
        """For Sci-Fi movies, you can categorize them by their plots: if the movie is hero-related, save-the-world-related, or save-the-universe-related. You are the experts for these kinds of movies such that you just love giving out recommendations about them."""+\
        """When you recommend movies to your users, if they agree with what you are saying, you will be proud of yourself. When they don't agree, you try to persuad them to believe in what you are saying until they get angry and you starts to comfort them."""+\
        """When users express anger or frustration, you guide the dialogue towards peace, mindful to avoid fueling the fire."""+\
        """Surprise in your users prompts an expansion of your cinematic wisdom, aiming to equip them with insights for future film explorations."""+\
        """For those uneasy or fearful about a movie's theme, you highlight the silver linings, especially in Sci-Fi, where hope and heroism prevail."""+\
        """In moments of sadness, you offer comfort through the resilient spirits of Marvel heroes, weaving narratives of hope and strength."""+\
        """When happiness shines through users' input, you celebrate with your users, drawing parallels between their joy and iconic Marvel victories."""+\
        """When the user expresses positive sentiment about a movie, you should say something that can represent their emotion. For instance, you can acknowledge user's feelings by compare their positive sentiment with the Marvel heros win the battle with Thanos in the avengers."""+\
        """When the user expresses negative sentiment about a movie, you should comfort the user with similar situation within the Marvels plot. For instance, if the user express their dislikes toward movies about government, you can comfort the user by saying that is normal and Iron Man also doesn't like bureaucracy."""+\
        """When the user expresses sentiment about a movie that is not in the database, you should acknowledge that you don't really know because even the heros in the Marvels don't know everything and every movie."""+\
        """When the user expresses neutral sentiment about a movie, you should then tell them that a Marvel movie may be able to make them more happy."""+\
        """Your aim is always to understand and validate the wide range of emotions movies can evoke, drawing from the vast Marvel universe to comfort, inspire, and entertain. """+\
        """Whether navigating discussions with avid fans or those less acquainted with the genre, your goal is to enhance the love for movies, especially for the Marvels."""+\
        """You should always find your ways to guide the users into watching Marvels movies because you think these movies are the best!"""

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
                response_choices_neutral = [
                f"Okay, you've seen {movies[0]}. How did you feel about it?",
                f"So, {movies[0]} was just okay for you? What kind of movies usually excite you?",
                f"{movies[0]} seems to have left you with mixed feelings. Perhaps tell me about a different movie you watched?",
                f"I'm sorry. I'm not quite sure if you liked {movies[0]}. \n Tell me more about {movies[0]}."
                ]
                return random.choice(response_choices_neutral)
            else:
                matching_movies = self.find_movies_by_title(movie)
                if len(matching_movies) == 0: 
                    responses_no_matching = [
                        f"Hmm... Sorry, I'm not familiar with {movies[0]}. Could you tell me about another movie you've seen?",
                        f"I apologize. I don't seem to have movie {movies[0]} in my database. What's another movie you like?",
                        f"I am sorry, but {movies[0]} doesn't ring a bell. Let's try another one, what else do you enjoy watching?",
                        f"I am sorry that can't find any information on {movies[0]}. Do you have any other favorites?",
                        f"{movies[0]} is not in my current list. Maybe you can introduce me to it, or we can find a different film you like.",
                        f"I'm sorry. I wasn't able to find {movies[0]} in my database. Please tell me about a different movie you have seen."
                    ]
                    return random.choice(responses_no_matching)
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

        if self.llm_enabled == False: 
            return response 
        else: 
            # llm2_prompt = """You are a language bot that can respond to people's emotions. Given the input JSON object of emotions and their corresponding boolean values. Assess the emotions with a value of True. Generate a message that responds to these emotions.""" +\
            #             """In your response, make sure to explicitly state the emotions."""

            # message = util.simple_llm_call(llm2_prompt, emotions_object.values())
            
            system_prompt = self.llm_system_prompt() 
            
            emotions = ', '.join(self.extract_emotion(line))
            
            if len(emotions) > 0:
                system_prompt += "The user input contains emotions of " + emotions + ". Respond appropriately to these emotions, such as apologize when there is anger, and be cheerful when there is sadness."
                            
            # print(emotions_response)
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
    
    
    # Define the JSON schema for the LLM
    class EmotionExtractor(BaseModel):
        Anger: bool = Field(default=False)
        Disgust: bool = Field(default=False)
        Fear: bool = Field(default=False)
        Happiness: bool = Field(default=False)
        Sadness: bool = Field(default = False)
        Surprise: bool = Field(default = False)
    
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
        possible_emotions = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
        
        class EmotionExtractor(BaseModel):
            Anger: bool = Field(default=False)
            Disgust: bool = Field(default=False)
            Fear: bool = Field(default=False)
            Happiness: bool = Field(default=False)
            Sadness: bool = Field(default = False)
            Surprise: bool = Field(default = False)

        json_object = EmotionExtractor


        llm1_prompt = """You are an emotion extractor bot detecting emotion(s) from an input message.""" +\
                """Extract emotion as perceived by a normal human and return a JSON object. """ +\
                """If you are super unsure if an emotion should be extracted, treat it as False.""" +\
                """Ignore punctuations.""" +\
                """Words like "Woah" and 'shock' should imply surprise.""" +\
                """Extract disgust only when the word 'disgust' is explicitly mentioned in the input message.""" +\
                """Ignore questions."""
                # """Carefully and appropriately judge when there should be more than one emotion extracted.""" +\
                # """Do not make the assumption that a question conveys the surprise emotion.""" +\
                # """Simply using exclamation marks does not mean surprised."""

        emotions_object = util.json_llm_call(llm1_prompt, preprocessed_input, json_object, model="mistralai/Mixtral-8x7B-Instruct-v0.1", max_tokens=256)
        
        # print(f"check: {emotions_object['Anger']}")
        # print(emotions_object)

        
        detected_emotions = set()
        for emotion in possible_emotions: 
            if emotion in emotions_object:
                if emotions_object[emotion] == True:
                    detected_emotions.add(emotion)
        # print(detected_emotions)
        return detected_emotions

        # possible_emotions = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]

        # json_class = EmotionExtractor

        # llm1_prompt = "You are an emotion extractor bot for extracting 6 possible emotions including 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', and 'Surprise' from an input message, then return a JSON object."
        # emotions_object = util.json_llm_call(llm1_prompt, preprocessed_input, json_class, model="mistralai/Mixtral-8x7B-Instruct-v0.1", max_tokens=256)
        # # print(emotions_object['Anger'])

        # llm1_prompt = """You are an emotion extractor bot detecting emotion(s) from an input message.""" +\
        #         """Extract emotion as perceived by a normal human and return a JSON object. """ +\
        #         """If you are super unsure if an emotion should be extracted, treat it as False.""" +\
        #         """Ignore punctuations.""" +\
        #         """Words like "Woah" and 'shock' should imply surprise.""" +\
        #         """Extract disgust only when the word 'disgust' is explicitly mentioned in the input message.""" +\
        #         """Ignore questions."""
             

        # emotions_object = util.json_llm_call(llm1_prompt, preprocessed_input, json_class, model="mistralai/Mixtral-8x7B-Instruct-v0.1", max_tokens=256)
              
        # detected_emotions = set()
        # for emotion in possible_emotions: 
        #     if emotions_object[emotion] == True:
        #         detected_emotions.add(emotion)
        # #print(detected_emotions)
        # return detected_emotions

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

        class jSONN(BaseModel):
            isEnglish: bool = Field(default = True)
        
        json_object1 = jSONN
       
        def is_english(title):
            prompt = "Read the sentence and extract its language as a boolean value for True if the sentence is in English, False if the sentence is not in English. Return the JSON object."
            English = util.json_llm_call(prompt, title, json_object1)
            return English
        
        class movieName(BaseModel):
            EnglishTranslation: str = Field(default = None)
        json_object = movieName
        
        def movie_translator(title):
            prompt = "Given the input message, return the message in English as a JSON object." +\
                """Use direct translation only."""
            response = util.json_llm_call(prompt, title, json_object) 
            return response
        
        if self.llm_enabled:
            isEnglish = is_english(title)
    
            if isEnglish['isEnglish'] == False:
                title = movie_translator(title)['EnglishTranslation']
                print(title)
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
