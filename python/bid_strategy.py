
# class bid_strategy must have following methods to be referenced:
# def __init__(self, camp_v):
# def bid(self, ctr):

# noticed that the true camp_v is 1000 * camp_v
class TruthfulBid:
    def __init__(self, camp_v):
        self.camp_v = camp_v

    def bid(self, ctr): # bid price < camp_v
        bid_price = int(self.camp_v * ctr * 1E3)
        return bid_price

class OptimalBid:
    def __init__(self, camp_v):
        self.camp_v = camp_v
        self.mu = 0

    def set_mu(self, mu):
        self.mu = mu
    
    def bid(self, ctr):
        bid_price = int(1 / (1 + self.mu) * self.camp_v * ctr * 1E3)
        return bid_price