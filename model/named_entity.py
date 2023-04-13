
class NamedEntity:
    """ A class that represents an extracted named entity from text """
    def __init__(self, start_index: int, end_index: int, tag: str):
        self.tag = tag
        self.start_index = start_index
        self.end_index = end_index

    def _to_string(self) -> str:
        str_result = "\n tag: " + str(self.tag) +\
                     "\n start: " + str(self.start_index) +\
                     "\n end: " + str(self.end_index)
        return str_result

    def __str__(self) -> str:
        return self._to_string()

    def __repr__(self) -> str:
        return self._to_string()
