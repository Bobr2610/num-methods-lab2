from math import exp, isfinite, log, sqrt


def f1(x1: float, x2: float, a: float) -> float:
    return x1 ** 2 / a ** 2 + x2 ** 2 / (a / 2) ** 2 - 1


def f2(x1: float, x2: float, a: float) -> float:
    return a * x2 - exp(x1) - x1


def phi(x1: float, x2: float, a: float) -> tuple[float, float]:
    x2 = a / 2 * sqrt(1 - (x1 / a) ** 2)
    argument = a * x2 - x1
    return log(argument), x2


def norm(first: tuple[float, float], second: tuple[float, float]) -> float:
    return max(abs(first[0] - second[0]), abs(first[1] - second[1]))


def iteration(
        x0: tuple[float, float], q: float, eps: float, a: float,
        max_iterations: int = 10000
) -> list[tuple[float, float]]:
    history = [x0]
    for _ in range(max_iterations):
        x_new = phi(*history[-1], a)
        history.append(x_new)
        if q / (1 - q) * norm(x_new, history[-2]) <= eps:
            return history

    raise ValueError("Итерации не сошлись")


def newton(
        x0: tuple[float, float], eps: float, a: float,
        max_iterations: int = 10000
) -> list[tuple[float, float]]:
    history = [x0]
    for _ in range(max_iterations):
        x1, x2 = history[-1]
        first, second = f1(x1, x2, a), f2(x1, x2, a)

        j11 = 2 * x1 / a ** 2
        j12 = 8 * x2 / a ** 2
        j21 = -exp(x1) - 1
        j22 = a
        determinant = j11 * j22 - j12 * j21
        if determinant == 0:
            raise ValueError("Нулевой определитель Якобиана")

        dx1 = (-first * j22 + j12 * second) / determinant
        dx2 = (first * j21 - j11 * second) / determinant
        x_new = (x1 + dx1, x2 + dx2)
        history.append(x_new)
        residual = max(abs(f1(*x_new, a)), abs(f2(*x_new, a)))
        if norm(x_new, history[-2]) <= eps and residual <= eps:
            return history

    raise ValueError("Ньютон не сошёлся")


def root(a: float, bounds: tuple) -> tuple[float, float]:
    def residual(x1: float) -> float:
        x2 = a / 2 * sqrt(1 - (x1 / a) ** 2)
        return f2(x1, x2, a)

    left, right = bounds[0]
    for _ in range(60):
        middle = (left + right) / 2
        if residual(left) * residual(middle) <= 0:
            right = middle
        else:
            left = middle

    x1 = (left + right) / 2
    return x1, a / 2 * sqrt(1 - (x1 / a) ** 2)


def report(name: str, history: list[tuple[float, float]]) -> None:
    x1, x2 = history[-1]
    print(f"{name}: ({x1:.10f}; {x2:.10f}), шагов: {len(history) - 1}")


def main() -> None:
    eps = float(input("Точность [0.01]: ") or 0.01)
    if not isfinite(eps) or eps <= 0:
        raise ValueError("Точность должна быть положительной и конечной")

    a = 3
    x0 = (1.125, 1.4)
    s = sqrt(1 - (1.25 / a) ** 2)
    q = max((1 + 1.25 / (2 * s)) / (a ** 2 / 2 * s - 1.25), 1.25 / (2 * a * s))
    report("Итерации", iteration(x0, q, eps, a))
    report("Ньютон", newton(x0, eps, a))


if __name__ == "__main__":
    main()
