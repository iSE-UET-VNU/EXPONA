import os
import logging
import random
import numpy as np

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

        self.model = self.get_model(
            provider=args.provider,
            model_name=args.model,
            temperature=getattr(args, "temperature", 1.0)
        )

    def get_model(self, provider, model_name, temperature) -> BaseChatModel:
        provider = provider.lower()
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def get_completion(self, user_prompt, n=1):
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ]

        return [self.model.invoke(messages).content for _ in range(n)]

    # def count_tokens_and_cost(self, user_prompt, completions, model_name=None):
    #     model_name = model_name or self.args.model
    #     input_text = f"{self.system_content}\n{user_prompt}"
    #     input_tokens = count_tokens(input_text, model_name=model_name)
    #     output_tokens = sum(count_tokens(c, model_name=model_name) for c in completions)

    #     price_table = {
    #         "gpt-4o": (2.5, 10.0),
    #         "gpt-4-1106-preview": (1.0, 3.0),
    #         "gpt-3.5-turbo": (0.5, 1.5),
    #         "claude-3-opus": (15.0, 75.0),
    #         "claude-3-sonnet": (3.0, 15.0),
    #         "claude-3-haiku": (0.25, 1.25),
    #         "gemini-pro": (0.1, 0.2),
    #     }

    #     input_price, output_price = price_table.get(model_name, (0, 0))

    #     cost = (input_tokens / 1e6) * input_price + (output_tokens / 1e6) * output_price

    #     return {
    #         "input_tokens": input_tokens,
    #         "output_tokens": output_tokens,
    #         "cost_usd": cost
    #     }


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

    def generate_feature_combinations(self, num_trials=200):
        k = int(self.p - np.sqrt(self.p))
            
        feature_combinations = set()
        while len(feature_combinations) < num_trials:
            comb = tuple(sorted(random.sample(range(self.p), k)))
            feature_combinations.add(comb)
        return [list(fc) for fc in feature_combinations]

    def fit_function(self, feature_idx, model):
        X = self.val_features[:, feature_idx]
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        model_key = model.lower()
        if model_key in self.model_dict:
            clf = self.model_dict[model_key]
            clf.fit(X, self.val_labels)
            return clf
        else:
            raise ValueError(f"Model '{model}' is not supported. Choose from: {list(self.model_dict.keys())}")

    def generate_lfs(self, model, num_trials):
        feature_combinations = self.generate_feature_combinations(num_trials=num_trials)
        lfs = []

        for comb in feature_combinations:
            selected_model = model
            lf = self.fit_function(comb, selected_model)
            lfs.append(lf)

        return lfs, feature_combinations