from otree.api import *
from .predict import *

import warnings
warnings.filterwarnings('ignore')

# __table_args__ = {'extend_existing': True}

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
    promise5 = models.FloatField(doc="Promise if sent 5", initial=0)
    promise10 = models.FloatField(doc="Promise if sent 10", initial=0)
    
    #sent_amount = models.FloatField()
    #sent_amount = models.FloatField(initial=float(bestDecision(promise5, promise10)))

    sent_back_amount = models.CurrencyField(doc="""Amount sent back by P""", min=cu(0))


class Player(BasePlayer):
    pass


# FUNCTIONS

def set_sent_amount(group: Group):
    sent_amount = bestDecision(group.promise5, group.promise10)
    return sent_amount

# def sent_back_amount_max(group: Group, sent_amount):
#     return sent_amount * C.MULTIPLIER


def set_payoffs(group: Group):
    sent_amount = set_sent_amount(group)
    p = group.get_player_by_id(1)
    p.payoff = sent_amount * C.MULTIPLIER - group.sent_back_amount

def update_results(group: Group):
    update_results_csv(group.promise5, group.promise10, set_sent_amount(group), group.sent_back_amount)


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
        sent_amount = set_sent_amount(group)

        tripled_amount = sent_amount * C.MULTIPLIER
        return dict(tripled_amount=tripled_amount, sent_amount=sent_amount)


class ResultsWaitPage(WaitPage):
    after_all_players_arrive = set_payoffs


class Results(Page):
    """This page displays the earnings of each player"""

    @staticmethod
    def vars_for_template(player: Player):
        group = player.group
        sent_amount = set_sent_amount(group)
        update_results(group)
        return dict(tripled_amount=sent_amount * C.MULTIPLIER, sent_amount=sent_amount)


page_sequence = [
    Introduction,
    Promise,
    SendBackWaitPage,
    SendBack,
    ResultsWaitPage,
    Results,
]