class Actor:
    def __init__(self, actor_id):
        self.id = actor_id
        pass

    def to_dict(self):
        return self.id


def create_random(actor_id):
    return Actor(actor_id)
