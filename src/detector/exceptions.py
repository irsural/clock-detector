class SearchingClockFaceError(Exception):
    """Raised when a clock face was not found"""

    def __init__(self, message='A clock face was not found'):
        self.message = message
        super().__init__(self.message)

class SearchingHandsError(Exception):
    """Raised when hands were not found"""

    def __init__(self, message='Hands of a clock were not found'):
        self.message = message
        super().__init__(self.message)