from math import exp, isfinite, log, sqrt
from pathlib import Path
from sys import argv


def f1(x1: float, x2: float, a: float) -> float:
    return x1 ** 2 / a ** 2 + x2 ** 2 / (a / 2) ** 2 - 1


def f2(x1: float, x2: float, a: float) -> float:
    return a * x2 - exp(x1) - x1


def phi(x1: float, x2: float, a: float) -> tuple[float, float]:
    if not all(isfinite(v) for v in (x1, x2, a)) or a <= 0 or abs(x1) > a:
        raise ValueError("Неверная область корня")
    x2 = a / 2 * sqrt(1 - (x1 / a) ** 2)
    argument = a * x2 - x1
    if argument <= 0:
        raise ValueError("Неположительный аргумент логарифма")
    return log(argument), x2


def norm(first: tuple[float, float], second: tuple[float, float]) -> float:
    return max(abs(first[0] - second[0]), abs(first[1] - second[1]))


def iteration(
        x0: tuple[float, float], q: float, eps: float, a: float,
        max_iterations: int = 10000
) -> list[tuple[float, float]]:
    if not isfinite(eps) or eps <= 0 or not 0 <= q < 1 or max_iterations <= 0:
        raise ValueError("Нужны eps > 0, 0 <= q < 1 и положительный лимит")
    history = [x0]
    for _ in range(max_iterations):
        x_new = phi(*history[-1], a)
        if not all(isfinite(v) for v in x_new):
            raise ValueError("Неконечное приближение")
        history.append(x_new)
        if q / (1 - q) * norm(x_new, history[-2]) <= eps:
            return history

    raise ValueError("Итерации не сошлись")


def newton(
        x0: tuple[float, float], eps: float, a: float,
        max_iterations: int = 10000
) -> list[tuple[float, float]]:
    if not isfinite(eps) or eps <= 0 or not isfinite(a) or a <= 0 or max_iterations <= 0:
        raise ValueError("Неверная точность, a или лимит")
    history = [x0]
    for _ in range(max_iterations):
        x1, x2 = history[-1]
        first, second = f1(x1, x2, a), f2(x1, x2, a)

        j11 = 2 * x1 / a ** 2
        j12 = 8 * x2 / a ** 2
        j21 = -exp(x1) - 1
        j22 = a
        determinant = j11 * j22 - j12 * j21
        if determinant == 0 or not isfinite(determinant):
            raise ValueError("Вырожденный или неконечный Якобиан")

        dx1 = (-first * j22 + j12 * second) / determinant
        dx2 = (first * j21 - j11 * second) / determinant
        x_new = (x1 + dx1, x2 + dx2)
        if not all(isfinite(v) for v in x_new):
            raise ValueError("Неконечное приближение")
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
    if not 0 < left < right < a or residual(left) * residual(right) > 0:
        raise ValueError("Неверный отрезок бисекции")
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
    if len(argv) > 2:
        raise ValueError("Запуск: python task2.py [input.txt]")
    path = Path(argv[1]) if len(argv) == 2 else Path(__file__).parent / 'data/task2.txt'
    eps, a, left1, right1, left2, right2 = map(float, path.read_text(encoding='utf-8-sig').split())
    if not all(isfinite(v) for v in (eps, a, left1, right1, left2, right2)):
        raise ValueError("Нужны конечные данные")
    if not 0 < left1 < right1 < a or not 0 < left2 < right2:
        raise ValueError("Неверные границы области")
    bounds = ((left1, right1), (left2, right2))
    x0 = ((left1 + right1) / 2, (left2 + right2) / 2)
    s = sqrt(1 - (right1 / a) ** 2)
    argument = a ** 2 / 2 * s - right1
    if argument <= 0:
        raise ValueError("Логарифм не определён на области")
    q = max((1 + right1 / (2 * s)) / argument, right1 / (2 * a * s))
    for x1 in (left1, right1):
        point = phi(x1, left2, a)
        if not all(bounds[i][0] <= point[i] <= bounds[i][1] for i in range(2)):
            raise ValueError("Итерации не сохраняют область")

    report("Итерации", iteration(x0, q, eps, a))
    report("Ньютон", newton(x0, eps, a))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, OverflowError) as error:
        raise SystemExit(str(error))
