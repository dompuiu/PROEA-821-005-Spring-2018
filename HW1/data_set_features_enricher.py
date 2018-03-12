class DataSetFeaturesEnricher:
    def __init__(self, data_set, features_method_names):
        self.data_set = data_set
        self.enriched_data_set = []
        self.features_method_names = features_method_names

    def get_enrich_data_set(self):
        for i, entry in enumerate(self.data_set, start=0):
            self.enriched_data_set.append(self.get_enriched_feature(entry))

        return self.enriched_data_set

    def get_enriched_feature(self, feature_vector):
        enriched_feature = []
        for method_name in self.features_method_names:
            new_value = getattr(self, method_name)(feature_vector[0])
            enriched_feature.append(new_value)

        enriched_feature.append(feature_vector[1])

        return enriched_feature

    def first_name_longer_that_last_name(self, v):
        return len(self.first_name(v)) > len(self.last_name(v))

    def has_middle_name(self, v):
        name_parts = v.split(' ')
        return len([name_part for name_part in name_parts if len(name_part.replace('.', '')) == 1]) > 0

    def first_name_starts_and_ends_with_same_letter(self, v):
        v = self.first_name(v)
        return len(v) > 1 and v[0].lower() == v[-1].lower()

    def first_name_come_alphabetically_before_their_last_name(self, v):
        first_name = self.first_name(v).lower()
        last_name = self.last_name(v).lower()

        return first_name[0] < last_name[0]

    def second_letter_of_their_first_name_a_vowel(self, v):
        first_name = self.first_name(v).lower()
        return first_name[1] in ['a', 'e', 'i', 'o', 'u']

    def is_the_number_of_last_name_letter_even(self, v):
        return len(self.last_name(v)) % 2 == 0

    def first_name(self, name):
        not_last_name = name.split(' ')[:-1]
        return " ".join([name_part for i, name_part in enumerate(not_last_name)
                         if i == 0 or len(name_part.replace('.', '')) != 1])

    def last_name(self, name):
        return name.split(' ')[-1]
