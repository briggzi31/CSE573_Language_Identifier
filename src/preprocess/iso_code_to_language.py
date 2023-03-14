import json
import pickle


class ISOCodeToLanguage:
    """
        Object mapping ISO codes to their respective English name
    """

    def __init__(self, iso_code_path="data/lan_to_language.json") -> None:
        with open(iso_code_path) as f:
            self.iso_to_lang = json.load(f)


    def __getitem__(self, item: str) -> str:
        """
        :param item: A string representing an ISO code
        :return: The name of the language mapped to the ISO code
        """
        if not isinstance(item, str):
            raise ValueError(f"not a str, but is {type(item)}")

        if not item in self.iso_to_lang:
            return None

        return self.iso_to_lang[item]


    def __str__(self) -> str:
        return str(self.iso_to_lang)


    def save(self, pickle_file_path):
        """
        :param output_file_path: The file path where the pickle file should be stored
            Saves the instance to a pickle file to be loaded in later :)
        :return: None
        """
        with open(pickle_file_path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    pickle_file_path = sys.argv[1]

    iso_to_lang = ISOCodeToLanguage()
    iso_to_lang.save(pickle_file_path)

