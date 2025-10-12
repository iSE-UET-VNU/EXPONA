configs = {
    "medabs": {
        "description": "medical abstracts into disease categories",
        "instructions": """
If the abstract is about Neoplasms, function returns 0.
If the abstract is about Digestive system diseases, function returns 1.
If the abstract is about Nervous system diseases, function returns 2.
If the abstract is about Cardiovascular diseases, function returns 3.
If the abstract is about General pathological conditions, function returns 4.
If the abstract cannot be categorized, function returns -1.

function signature: def label_function(abstract)
"""
    },
    "agnews": {
        "description": "news article headlines and descriptions into topic categories",
        "instructions": """
If the article is about World, function returns 0.
If the article is about Sports, function returns 1.
If the article is about Business, function returns 2.
If the article is about Science/Technology, function returns 3.
If the article cannot be categorized, function returns -1.

function signature: def label_function(text)
"""
    },
    "sms": {
        "description": "SMS messages into spam or ham (non-spam)",
        "instructions": """
If the message is legitimate (ham), function returns 0.
If the message is spam, function returns 1.
If the message cannot be categorized, function returns -1.

function signature: def label_function(message)
"""
    },
    "imdb": {
        "description": "movie reviews into positive or negative sentiment",
        "instructions": """
If the review expresses negative sentiment, function returns 0.
If the review expresses positive sentiment, function returns 1.
If the review cannot be categorized, function returns -1.

function signature: def label_function(review)
"""
    },
    "youtube": {
        "description": "YouTube comments into spam or ham (non-spam)",
        "instructions": """
If the comment is legitimate (ham), function returns 0.
If the comment is spam, function returns 1.
If the comment cannot be categorized, function returns -1.

function signature: def label_function(comment)
"""
    },
    "yelp": {
        "description": "Yelp reviews into positive or negative sentiment",
        "instructions": """
If the review expresses negative sentiment, function returns 0.
If the review expresses positive sentiment, function returns 1.
If the review cannot be categorized, function returns -1.

function signature: def label_function(review)
"""
    },
    "clickbait": {
    "description": "article headlines classified as clickbait or not",
    "instructions": """
If the headline is not clickbait, function returns 0.
If the headline is clickbait, function returns 1.
If the headline cannot be categorized, function returns -1.

function signature: def label_function(headline)
"""
},
    "finance": {
    "description": "sentences from economic/financial texts into sentiment categories",
    "instructions": """
If the sentence expresses negative economic sentiment, function returns 0.
If the sentence is neutral or factual without clear sentiment, function returns 1.
If the sentence expresses positive economic sentiment, function returns 2.
If the sentence cannot be categorized, function returns -1.

function signature: def label_function(sentence)
"""
},
    "tos": {
    "description": "Terms of Service sentences into types of unfair contractual terms",
    "instructions": """
If the sentence includes a clause limiting the platform's liability, function returns 0.
If the sentence allows the platform to unilaterally terminate the service, function returns 1.
If the sentence allows the platform to unilaterally change the contract terms, function returns 2.
If the sentence allows the platform to remove or block user content, function returns 3.
If the sentence implies that a contract is formed simply by using the service, function returns 4.
If the sentence chooses a specific national law to govern the contract, function returns 5.
If the sentence assigns disputes to a specific court or jurisdiction, function returns 6.
If the sentence requires arbitration instead of court proceedings, function returns 7.
If the sentence includes other types of potentially unfair terms not listed above, function returns 8.
If the sentence cannot be categorized, function returns -1.

function signature: def label_function(sentence)
"""
},
    "chemprot": {
    "description": "chemical-protein relation classification from biomedical abstracts",
    "instructions": """
If the sentence describes a 'Part of' relationship between a chemical and a protein, function returns 0.
If the chemical acts as a regulator of the protein, function returns 1.
If the chemical upregulates the protein, function returns 2.
If the chemical downregulates the protein, function returns 3.
If the chemical is an agonist for the protein, function returns 4.
If the chemical is an antagonist of the protein, function returns 5.
If the chemical modulates the protein's function, function returns 6.
If the chemical serves as a cofactor to the protein, function returns 7.
If the chemical is a substrate or product in a reaction involving the protein, function returns 8.
If the sentence does not express any valid relation between chemical and protein, function returns 9.
If the relation cannot be determined, function returns -1.

function signature: def label_function(sentence)
"""
},
    "massive": {
    "description": "English utterances for intent classification in intelligent voice assistant",
    "instructions": """
If the utterance is related to alarm setting or querying, function returns 0.
If the utterance concerns audio settings or control, function returns 1.
If the utterance is about Internet of Things (IoT) devices, function returns 2.
If the utterance involves calendar events, function returns 3.
If the utterance is a request to play media, function returns 4.
If the utterance is a general request or unclear, function returns 5.
If the utterance relates to dates or times, function returns 6.
If the utterance concerns food ordering or takeaway, function returns 7.
If the utterance is about news, function returns 8.
If the utterance is about music or music playback, function returns 9.
If the utterance is about the weather, function returns 10.
If the utterance is a factual question (Q&A), function returns 11.
If the utterance involves social media or messaging, function returns 12.
If the utterance requests a recommendation (e.g. restaurants, music), function returns 13.
If the utterance is about cooking or recipes, function returns 14.
If the utterance involves transportation or travel, function returns 15.
If the utterance concerns email, function returns 16.
If the utterance involves task or shopping lists, function returns 17.
If the utterance cannot be categorized, function returns -1.

function signature: def label_function(utterance)
"""
},
}