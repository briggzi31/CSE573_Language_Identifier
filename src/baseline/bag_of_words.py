import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

class BOW:

    BATCH_SIZE = 100

    def __init__(self, train_data_path):
        self.train_data_path = train_data_path
        print("Processing training data...")
        self.train_data, self.doc = self.process_data(train_data_path)
        print("Vectorizing data...")
        self.count_vectors = self.count_vectorize()
        return

    def main(self):
        return

    def process_data(self, train_data_path):
        df = pd.read_csv(train_data_path)
        document = df['sentence']
        return df, document

    def count_vectorize(self):
        print("Fitting train data to vectorizer...")
        cv.fit(self.doc)
        vocab = cv.vocabulary_
        print("Size of vocabulary is " + str(len(vocab)))  # 2007959
        print("Transforming data to vectors...")
        feature_vectors = self.batch_process(self.doc, self.BATCH_SIZE)
        print(feature_vectors[0:5])
        return feature_vectors

    def batch_process(self, doc, BATCH_SIZE):
        print("Batch processing...")
        feature_vectors = []
        for batch in self.chunks(doc, BATCH_SIZE):
            vectors = cv.transform(batch)
            batch_done = vectors.toarray()
            feature_vectors.extend(batch_done)
        return feature_vectors

    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def tfidf_vectorize(self):
        return


if __name__ == '__main__':
    bow = BOW('../../data/train.csv')


# document = ["I was late to school.", "レクシイさんの名前は何ですか？", "今天真的好累啊。",
#             "Oh, jetzt ist es wirklich verwirrend...", "Bebo los vientos por ti.",
#             "Ne sous-estime pas ma puissance.", "Я бы так и сказал.", "Công lý rất đắt.",
#             "את נראית אפיל יפה יותר ממה שאני זוכ."]
#
# cv.fit(document)
#
# print("Vocabulary: ", cv.vocabulary_)
#
# print(type(document))
# vector = cv.transform(document)
# print(type(vector))
# print(vector.toarray())
