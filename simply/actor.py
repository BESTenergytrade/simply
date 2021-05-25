class Actor:
    def __init__(self, actor_id):
        self.id = actor_id
        pass


def create_random(actor_id):
    return Actor(actor_id)
