from otree.api import *
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.metrics import \
    r2_score, get_scorer
from sklearn.preprocessing import \
    StandardScaler, PolynomialFeatures

import warnings
warnings.filterwarnings('ignore')

doc = """
This is a standard 2-player trust game where the amount sent by player 1 gets
tripled. The trust game was first proposed by
<a href="http://econweb.ucsd.edu/~jandreon/Econ264/papers/Berg%20et%20al%20GEB%201995.pdf" target="_blank">
    Berg, Dickhaut, and McCabe (1995)
</a>.
"""


class C(BaseConstants):
    NAME_IN_URL = 'trustlending'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1
    INSTRUCTIONS_TEMPLATE = 'trustlending/instructions.html'
    # Initial settings
    ENDOWMENT = cu(10)
    MULTIPLIER = 3



class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    promise5 = models.CurrencyField(doc="""Amount promised to send in exchange for 5""", min=cu(0))
    promise10 = models.CurrencyField(doc="""Amount promised to send in exchange for 10""", min=cu(0))
    
    # load data into df
    url = "https://raw.githubusercontent.com/nandikachirala/trust-lending-AI/main/ArtificialData-For-Algorithm.csv"
    dataset = pd.read_csv(url)

    # forming two models
    datasetFive = dataset.loc[dataset['AmountSent'] == 5]
    datasetTen = dataset.loc[dataset['AmountSent'] == 10]
    dataFive = datasetFive.values
    dataTen = datasetTen.values

    if(promise5 != 0):
        made5 = 1
    else:
        made5 = 0
    if(promise10 != 0):
        made10 = 1
    else:
        made10 = 0

    # Model for 5

    Xfive, yfive = dataFive[:, :-1], dataFive[:, -1]
    model = Lasso(alpha=1.0)
    # define model evaluation method
    cvFive = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, Xfive, yfive, scoring='r2', cv=cvFive, n_jobs=-1)
    # force scores to be positive
    scores = np.absolute(scores)
    #print('Mean Absolute Error: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    # tune alpha for 5
    modelFive = LassoCV(alphas=np.arange(0, 1, 0.01), cv=cvFive, n_jobs=-1)
    modelFive.fit(Xfive, yfive)
    #print('alpha: %f' % modelFive.alpha_)

    # predict using tuned alpha
    modelFive = Lasso(alpha=0.18)
    # fit model
    modelFive.fit(Xfive, yfive)
    # new fake data
    row = [promise5, promise5, 5, made5, made10]
    # make a prediction
    predictFive = modelFive.predict([row])
    # summarize prediction
    #print('Predicted: %.3f' % predictFive)
    #print(modelFive.coef_)
    #print(modelFive.intercept_)

    # Model for 10

    Xten, yten = dataTen[:, :-1], dataTen[:, -1]

    modelTen = Lasso(alpha=1.0)
    # define model evaluation method
    cvTen = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(modelTen, Xten, yten, scoring='r2', cv=cvTen, n_jobs=-1)
    # force scores to be positive
    scores = np.absolute(scores)
    #print('Mean Absolute Error: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    # tune alpha for 10
    from sklearn.linear_model import LassoCV
    modelTen = LassoCV(alphas=np.arange(0, 1, 0.01), cv=cvTen, n_jobs=-1)
    modelTen.fit(Xten, yten)
    #print('alpha: %f' % modelTen.alpha_)

    # predict using tuned alpha
    modelTen = Lasso(alpha=0.03)
    # fit model
    modelTen.fit(Xten, yten)
    # new fake data
    row = [promise5, promise10, 10, made5, made10]
    # make a prediction
    predictTen = modelTen.predict([row])
    # summarize prediction
    #print('Predicted: %.3f' % predictTen)
    #print(modelTen.coef_)
    #print(modelTen.intercept_)

    # Choosing send amount
    predictFive = float(predictFive) + 5
    predictTen = float(predictTen)
    outcomeMap = {10:0, predictFive:5, predictTen:10}
    bestDecision = max(10, predictFive, predictTen)
    #print(bestDecision)
    #TODO: add chosen decision row into file and re-evaluate
    sent_amount = outcomeMap[bestDecision]

    sent_back_amount = models.CurrencyField(doc="""Amount sent back by P""", min=cu(0))


class Player(BasePlayer):
    pass


# FUNCTIONS

def sent_back_amount_max(group: Group):
    return group.sent_amount * C.MULTIPLIER


def set_payoffs(group: Group):
    p = group.get_player_by_id(1)
    p.payoff = group.sent_amount * C.MULTIPLIER - group.sent_back_amount


# PAGES
class Introduction(Page):
    pass


class Promise(Page):
    """P2 needs to make a promise to P1 (the algorithm) regarding how much they are going to return"""

    form_model = 'group'
    form_fields = ['promise5', 'promise10']
    #sent_amount = best_decision

    @staticmethod
    def is_displayed(player: Player):
        return player.id_in_group == 1



class SendBackWaitPage(WaitPage):
    pass


class SendBack(Page):
    """This page is only for P2
    P2 sends back some amount (of the tripled amount received) to P1"""

    form_model = 'group'
    form_fields = ['sent_back_amount']

    @staticmethod
    def is_displayed(player: Player):
        return player.id_in_group == 1

    @staticmethod
    def vars_for_template(player: Player):
        group = player.group

        tripled_amount = group.sent_amount * C.MULTIPLIER
        return dict(tripled_amount=tripled_amount)


class ResultsWaitPage(WaitPage):
    after_all_players_arrive = set_payoffs


class Results(Page):
    """This page displays the earnings of each player"""

    @staticmethod
    def vars_for_template(player: Player):
        group = player.group

        return dict(tripled_amount=group.sent_amount * C.MULTIPLIER)


page_sequence = [
    Introduction,
    Promise,
    SendBackWaitPage,
    SendBack,
    ResultsWaitPage,
    Results,
]