import pandas as pd
from itertools import combinations
import itertools
from scipy import misc

class AssociationRules():

    def __init__(self):
        #def __init__(self, max_itemsets_per_type, combo_card=(2, 2)):
        # self.max_itemsets_per_type = max_itemsets_per_type
        # self.combo_card = combo_card
        print "initializing..."

    def fit(self, data):
        print "gathering good itemsets..."
        df = data.groupby(['VisitNumber', 'TripType'])['Upc'].apply(frozenset).reset_index()
        # common_itemsets = self.get_common_itemsets(self.max_itemsets_per_type, df)
        # self.combinations = self.get_itemset_combos(common_itemsets, self.combo_card)
        self.combinations = self.get_itemset_ngrams(2, df)

    def get_common_itemsets(self, n_itemsets, full_data):
        df = full_data.groupby(['Upc', 'TripType'])['VisitNumber'].count().reset_index()
        common_itemsets = df.sort_values('VisitNumber', ascending=False)
        ci = common_itemsets.groupby('TripType').head(n_itemsets)
        ci = ci[ci['VisitNumber'] >= 10]
        return ci['Upc']

    def get_itemset_ngrams(self, n, full_data):
        df = full_data.groupby(['Upc', 'TripType'])['VisitNumber'].count().reset_index()
        n_grams = df[df['Upc'].apply(lambda x: len(x) == n)].sort_values(['VisitNumber'], ascending=False)
        return n_grams['Upc'].drop_duplicates().iloc[:100]


    def get_itemset_combos(self, itemsets, combo_len):
        itemsets = frozenset().union(*itemsets.tolist())
        print "Anticipated Number of Itemsets: {}".format(misc.comb(len(itemsets), combo_len[1]))
        combos = itertools.chain(*[combinations(itemsets, i) for i in range(combo_len[0], combo_len[1] + 1)])
        combos = map(frozenset, combos)
        return combos

    def transform(self, data):
        print "computing rules..."
        df = data.groupby(['VisitNumber', 'TripType'])['Upc'].apply(frozenset).reset_index()
        data = df.groupby(['Upc', 'TripType'])['VisitNumber'].count().reset_index()
        best_itemsets = pd.DataFrame(columns=['ItemSet', 'VisitCount', 'TripType', 'Confidence'])
        count = 1
        for i in self.combinations:
            if (count % 100 == 0):
                print "Checking Itemset {}...".format(count)
            count = count + 1
            # get only visits that included this itemset
            df03 = data[data['Upc'].apply(lambda x: i.issubset(x))]
            # if there are no supersets, then this is an empty dataset. Pass
            if (df03.empty == True):
                continue
            # get total number of visits that included this itemset
            sc = df03['VisitNumber'].sum()
            df03 = df03.rename(columns={'VisitNumber': 'Confidence'})
            superset_count = float(sc)
            # figure ratio of items bought per trip classification, put it into a new df
            classified_superset_confidence= df03.groupby(['TripType'])['Confidence'].agg(
                lambda x: x.sum() / superset_count).reset_index()
            best = classified_superset_confidence.loc[classified_superset_confidence['Confidence'].idxmax()][
                ['TripType', 'Confidence']].to_dict()
            best['VisitCount'] = sc
            best['ItemSet'] = i
            best_itemsets = best_itemsets.append(best, ignore_index=True)

        return best_itemsets

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

