def build_scheduler(optimizer, warmup_steps: int = 0):
    if warmup_steps <= 0:
        return None

    def lr_lambda(step):
        return min((step + 1) / warmup_steps, 1.0)

    return __import__("torch").optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda
    )
