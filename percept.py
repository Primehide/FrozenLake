class Percept:
    def __init__(self, current_state: object, action: float, reward: float, next_state: int, final_state: bool):
        self._current_state = current_state
        self._action = action
        self._reward = reward
        self._next_state = next_state
        self._final_state = final_state

    @property
    def current_state(self) -> object:
        return self._current_state

    @property
    def action(self) -> float:
        return self._action

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def next_state(self) -> int:
        return self._next_state

    @property
    def final_state(self) -> bool:
        return self._final_state


