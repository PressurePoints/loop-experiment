
# noticed that the true camp_v is 1000 * camp_v
class TruthfulBid:
    def __init__(self, camp_v):
        self.camp_v = camp_v

    def bid(self, ctr): # bid price < camp_v
        bid_price = int(self.camp_v * ctr * 1E3)
        return bid_price