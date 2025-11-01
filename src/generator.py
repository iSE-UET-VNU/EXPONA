import os
import logging

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

load_dotenv()
logger = logging.getLogger(__name__)

class LLMGenerator:
    def __init__(self, system_prompt, args):
        self.args = args
        self.system_prompt = system_prompt

        self.model = self.get_model()

    def get_model(self) -> BaseChatModel:
        provider = self.args.llm_model.split("/")[0]
        model = self.args.llm_model.split("/")[-1]
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                api_key=self.args.api_key or os.getenv("OPENAI_API_KEY"),
            )
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=self.args.api_key or os.getenv("GOOGLE_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def get_completion(self, user_prompt, n=1):
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ]

        return [self.model.invoke(messages) for _ in range(n)]

    def calculate_cost(self, input_tokens, output_tokens):
        price_table = {
            "gpt-4o": (2.5, 10.0),
            "gpt-3.5-turbo": (0.5, 1.5),
            "gpt-4.1": (2.5, 10.0),
            "gemini-2.5-flash": (0.1, 0.2),
        }

        input_price, output_price = price_table.get(self.args.model, (2.5, 10))

        cost = (input_tokens / 1e6) * input_price + (output_tokens / 1e6) * output_price

        return cost


class ScikitGenerator:
    def __init__(self, val_features, val_labels):
        self.val_features = val_features
        self.val_labels = val_labels
        self.p = self.val_features.shape[1]

        self.model_dict = {
            'dt': DecisionTreeClassifier(),
            'lr': LogisticRegression(),
            'knn': KNeighborsClassifier(),
            'svm': SVC(probability=True),
            'mlp': MLPClassifier()
        }

    def fit_function(self, model):
        X = self.val_features
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        model_key = model.lower()
        if model_key in self.model_dict:
            clf = self.model_dict[model_key]
            clf.fit(X, self.val_labels)
            return clf
        else:
            raise ValueError(f"Model '{model}' is not supported. Choose from: {list(self.model_dict.keys())}")

    def generate_lfs(self, model):
        selected_model = model
        lf = self.fit_function(selected_model)

        return lf