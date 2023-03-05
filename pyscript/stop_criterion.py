def stop_by_time(T):
    def __fn__(duration = 0, accuracy = 0, steps = 0):
        return (duration > T)
    return __fn__

def stop_by_step(T):
    def __fn__(duration = 0, accuracy = 0, steps = 0):
        return (steps > T)
    return __fn__

def stop_by_acc(T):
    def __fn__(duration = 0, accuracy = 0, steps = 0):
        return (accuracy > T)
    return __fn__