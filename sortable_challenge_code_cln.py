###Code by Mengyin Joy Wang
###solution to Sortable Coding Challenge using a combination of tfidf, lsi
###and a naive matching score

import pandas as pd
import numpy as np
import json
import math
import re
import gensim
import argparse
from collections import defaultdict

class ListingsMatcher():

    def __init__(self, listings_df=pd.DataFrame(), products_df=pd.DataFrame()):
        self.listings_df = listings_df
        self.products_df = products_df
        self.match_score_threshold = 1
        self.no_match_marker = "|--NO MATCH--|"

    def read_data(self, listings_fname = "listings.txt", products_fname="products.txt"):
        #read listings and products into dataframes
        listings_f = open(listings_fname)
        listings = [json.loads(line, encoding='utf-8') for line in listings_f]
        listings_f.close()
        self.listings_df = pd.DataFrame(listings)

        products_f = open(products_fname)
        products = [json.loads(line, encoding='utf-8') for line in products_f]
        products_f.close()
        self.products_df = pd.DataFrame(products)

    def data_cln_and_tokenize(self):
        #do some data cleaning and tokenizing
        self.products_df['product_name_cln'] = self.products_df['product_name'].apply(lambda x: re.sub("_|-"," ",
                                                                                   re.sub("\(|\)", "", x.lower())))
        #Note: re.sub("_|-"," ",x) may be more helpful?

        self.products_df['prod_texts'] = self.products_df['product_name_cln'].apply(lambda name: name.split(" "))
        self.products_df.apply(lambda row: row['prod_texts'].extend([row['family'].lower() if pd.notnull(row['family'])\
                                                                else row['prod_texts'][0],
                                                                row['manufacturer'].lower() if pd.notnull(
                                                                    row['manufacturer']) else row['prod_texts'][0],
                                                                row['model'].lower() if pd.notnull(row['model'])\
                                                                    else row['prod_texts'][0]]),
                          axis=1)
        self.products_df['prod_texts'] = self.products_df['prod_texts'].apply(lambda txts: list(set(txts)))

        self.listings_df['listings_texts'] = self.listings_df['title'].apply(lambda name: re.sub("_|\(|\)|-", "",
                                                                                     name.lower()).split(" "))
        self.listings_df.apply(lambda row: row['listings_texts'].extend(row['manufacturer'].lower().split(" ")),
                          axis=1)

    def analysis_and_matching(self):
        #create corpora and dictionary for semantic analysis
        product_texts =[l for l in self.products_df['prod_texts']]
        product_dictionary = gensim.corpora.Dictionary(product_texts)

        #products_df = products_df.drop('vec', 1)
        self.products_df['prod_vec'] = self.products_df['prod_texts'].apply(lambda txts: product_dictionary.doc2bow(txts))

        ##transformation via tfidf
        product_corpus =[l for l in self.products_df['prod_vec']]

        tfidf = gensim.models.TfidfModel(product_corpus, wlocal=math.log1p,
                                         wglobal=lambda doc_freq, total_docs: (1.0 * total_docs / doc_freq))
        prod_corpus_tfidf = tfidf[product_corpus]

        #self.products_df['prod_vec_tfidf'] = self.products_df['prod_vec'].apply(lambda vec: tfidf[vec])

        #have reasonable num_topic
        prod_topic_num = (len(set(self.products_df['manufacturer']))+len(set(self.products_df['family']))+
                          len(set(self.products_df['model'])))/3

        #latent semantic analysis
        lsi = gensim.models.LsiModel(prod_corpus_tfidf, id2word=product_dictionary, num_topics=prod_topic_num)
        prod_corpus_lsi = lsi[prod_corpus_tfidf]

        #apply vector transforms to listings
        self.listings_df['listings_vec'] = self.listings_df['listings_texts'].apply(lambda txts: product_dictionary.doc2bow(txts))
        self.listings_df['listings_vec_tfidf'] = self.listings_df['listings_vec'].apply(lambda vec: tfidf[vec])
        self.listings_df['listings_vec_lsi'] = self.listings_df['listings_vec_tfidf'].apply(lambda vec: lsi[vec])

        #create index of similarities
        prod_index = gensim.similarities.MatrixSimilarity(prod_corpus_lsi,num_features=prod_topic_num)

        def find_lsi_match(vec_lsi): #scoring using just lsi
            prod_sims = prod_index[vec_lsi]
            prod_sims= sorted(enumerate(prod_sims), key=lambda item: -item[1])
            return self.products_df.ix[prod_sims[0][0]]['product_name']

        def naive_score_list(listing_texts): #scoring using a naive matching count
            temp_score = self.products_df['prod_texts'].apply(lambda prod_texts: len(set.intersection(set(prod_texts),
                                                                            set(listing_texts))))
            return temp_score/np.sum(temp_score)

        #scoring using a weighted mix of both metrics
        def find_lsi_with_naive_score(txts, vec_lsi):
            prod_sims = prod_index[vec_lsi]
            naive_scores = self.products_df['prod_texts'].apply(lambda prod_texts: len(set.intersection(set(prod_texts),
                                                                            set(txts))))
            #naive_scores = naive_scores/np.sum(naive_scores)
            weighted_scores = [prod_sims[i]*naive_scores[i] for i in xrange(len(prod_sims))]
            weighted_scores = sorted(enumerate(weighted_scores), key=lambda item: -item[1])
            match = self.products_df.ix[weighted_scores[0][0]]['product_name']\
                    if weighted_scores > self.match_score_threshold else self.no_match_marker
            return match

        self.listings_df['prod_match_weighted'] = self.listings_df.apply(lambda row: find_lsi_with_naive_score(
                                                                        row['listings_texts'],
                                                                        row['listings_vec_lsi']),
                                                             axis=1)

    def write_output(self, out_fname="results.txt"):
        #write output to file
        listings_grouped = self.listings_df[['manufacturer','title','price','currency',
                                        'prod_match_weighted']].groupby('prod_match_weighted')

        results_file = open(out_fname, "w")
        for prod_name, listing_grp in listings_grouped:
            if prod_name!=self.no_match_marker:
                jsonData = {"product_name": prod_name,
                            "listings": [{'title': row[1]['title'],
                                          'manufacturer': row[1]['manufacturer'],
                                          'currency': row[1]['currency'],
                                          'price': row[1]['price']} for row in listing_grp.iterrows()]}
                json.dump(jsonData, results_file, ensure_ascii=True)
                results_file.write("\n")
        results_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("listings_file", type=str, help="name of listings file", default="listings.txt")
    parser.add_argument("products_file", type=str, help="name of products file", default="products.txt")
    parser.add_argument("out_file", type=str, help="name of desired output file", default="results.txt")
    args = parser.parse_args()

    matcher = ListingsMatcher()
    matcher.read_data(args.listings_file, args.products_file)
    matcher.data_cln_and_tokenize()
    matcher.analysis_and_matching()
    matcher.write_output(args.out_file)
