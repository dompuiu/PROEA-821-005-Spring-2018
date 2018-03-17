class DataSetFeaturesEnricher:
    def __init__(self, data_set, features_method_names):
        self.data_set = data_set
        self.enriched_data_set = []
        self.features_method_names = features_method_names

    @staticmethod
    def first_name(name):
        not_last_name = name.split(' ')[:-1]
        return " ".join([name_part for i, name_part in enumerate(not_last_name)
                         if i == 0 or len(name_part.replace('.', '')) != 1])

    @staticmethod
    def last_name(name):
        return name.split(' ')[-1]

    @staticmethod
    def first_name_longer_that_last_name(v):
        return len(DataSetFeaturesEnricher.first_name(v)) > len(DataSetFeaturesEnricher.last_name(v))

    @staticmethod
    def has_middle_name(v):
        name_parts = v.split(' ')
        return len([name_part for name_part in name_parts if len(name_part.replace('.', '')) == 1]) > 0

    @staticmethod
    def first_name_starts_and_ends_with_same_letter(v):
        v = DataSetFeaturesEnricher.first_name(v)
        return len(v) > 1 and v[0].lower() == v[-1].lower()

    @staticmethod
    def first_name_come_alphabetically_before_their_last_name(v):
        first_name = DataSetFeaturesEnricher.first_name(v).lower()
        last_name = DataSetFeaturesEnricher.last_name(v).lower()

        return first_name[0] < last_name[0]

    @staticmethod
    def second_letter_of_their_first_name_a_vowel(v):
        first_name = DataSetFeaturesEnricher.first_name(v).lower()
        return first_name[1] in ['a', 'e', 'i', 'o', 'u']

    @staticmethod
    def is_the_number_of_last_name_letters_even(v):
        return len(DataSetFeaturesEnricher.last_name(v)) % 2 == 0

    @staticmethod
    def name_shorter_that_15_chars(v):
        return len(v) <= 15

    @staticmethod
    def name_formed_from_more_than_two_words(v):
        return len(v.split(' ')) > 2

    @staticmethod
    def last_letter_is_vowel(v):
        return v[-1] in ['a', 'e', 'i', 'o', 'u']

    @staticmethod
    def number_of_vowels_is_odd(v):
        c = 0
        for letter in v:
            if letter in ['a', 'e', 'i', 'o', 'u']:
                c += 1
        return c % 2 == 1

    def get_enrich_data_set(self):
        for i, entry in enumerate(self.data_set, start=0):
            self.enriched_data_set.append(self.get_enriched_feature(entry))

        return self.enriched_data_set

    def get_enriched_feature(self, feature_vector):
        enriched_feature = []
        for method_name in self.features_method_names:
            new_value = getattr(DataSetFeaturesEnricher, method_name)(feature_vector[0])
            enriched_feature.append(new_value)

        enriched_feature.append(feature_vector[1])

        return enriched_feature
