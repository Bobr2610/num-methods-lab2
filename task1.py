from math import isfinite


def f(x: float) -> float:
    return x ** 3 - 2 * x ** 2 - 10 * x + 15


def df(x: float) -> float:
    return 3 * x ** 2 - 4 * x - 10


def d2f(x: float) -> float:
    return 6 * x - 4


def phi(x: float, lam: float) -> float:
    return x - lam * f(x)


def iteration(
        x0: float, q: float, eps: float, lam: float,
        max_iterations: int = 10000
) -> list[float]:
    history = [x0]
    for _ in range(max_iterations):
        x_new = phi(history[-1], lam)
        history.append(x_new)
        if q / (1 - q) * abs(x_new - history[-2]) <= eps:
            return history

    raise ValueError("Итерации не сошлись")


def start(a: float, b: float) -> float:
    for x in (a, b):
        if f(x) == 0 or f(x) * d2f(x) > 0:
            return x
    raise ValueError("Нет подходящего начального приближения")


def newton(
        x0: float, eps: float, max_iterations: int = 10000
) -> list[float]:
    history = [x0]
    for _ in range(max_iterations):
        x = history[-1]
        if df(x) == 0:
            raise ValueError("Нулевая производная")
        x_new = x - f(x) / df(x)
        history.append(x_new)
        if abs(x_new - x) <= eps and abs(f(x_new)) <= eps:
            return history

    raise ValueError("Ньютон не сошёлся")


def root(a: float, b: float) -> float:
    for _ in range(60):
        middle = (a + b) / 2
        if f(a) * f(middle) <= 0:
            b = middle
        else:
            a = middle
    return (a + b) / 2


def report(name: str, history: list[float]) -> None:
    print(f"{name}: {history[-1]:.10f}, шагов: {len(history) - 1}")


def main() -> None:
    eps = float(input("Точность [0.01]: ") or 0.01)
    if not isfinite(eps) or eps <= 0:
        raise ValueError("Точность должна быть положительной и конечной")

    for a, b, lam, q in (
            (1, 1.5, -1 / 11, 7 / 44),
            (3.5, 4, 1 / 22, 37 / 88)
    ):
        print(f"[{a:g}; {b:g}]")
        report("Итерации", iteration((a + b) / 2, q, eps, lam))
        report("Ньютон", newton(start(a, b), eps))


if __name__ == "__main__":
    main()
