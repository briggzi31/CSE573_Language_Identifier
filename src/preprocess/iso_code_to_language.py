import json


class ISOCodeToLanguage:
    """
        Object mapping ISO codes to their respective English name
    """

    def __init__(self):
        with open("data/lan_to_language.json") as f:
            self.iso_to_lang = json.load(f)


    def __getitem__(self, item):
        """
        :param item: A string representing an ISO code
        :return: The name of the language mapped to the ISO code
        """
        if not isinstance(item, str):
            raise ValueError(f"not a str, but is {type(item)}")

        if not item in self.iso_to_lang:
            return None

        return self.iso_to_lang[item]


    def __str__(self):
        return str(self.iso_to_lang)


if __name__ == '__main__':
    iso_to_lang = ISOCodeToLanguage()

    print(iso_to_lang)
    print(iso_to_lang["eng"])
