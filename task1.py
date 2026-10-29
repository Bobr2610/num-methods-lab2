from math import isfinite
from pathlib import Path
from sys import argv


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
    if not isfinite(eps) or eps <= 0 or not 0 <= q < 1 or lam == 0 or max_iterations <= 0:
        raise ValueError("Нужны eps > 0, 0 <= q < 1 и положительный лимит")
    history = [x0]
    for _ in range(max_iterations):
        x_new = phi(history[-1], lam)
        if not isfinite(x_new):
            raise ValueError("Неконечное приближение")
        history.append(x_new)
        if q / (1 - q) * abs(x_new - history[-2]) <= eps:
            return history

    raise ValueError("Итерации не сошлись")


def start(a: float, b: float) -> float:
    if not 0 < a < b or f(a) * f(b) > 0 or df(a) * df(b) <= 0 or d2f(a) * d2f(b) <= 0:
        raise ValueError("Не выполнены условия Ньютона на отрезке")
    for x in (a, b):
        if f(x) == 0 or f(x) * d2f(x) > 0:
            return x
    raise ValueError("Нет подходящего начального приближения")


def newton(
        x0: float, eps: float, max_iterations: int = 10000
) -> list[float]:
    if not isfinite(eps) or eps <= 0 or max_iterations <= 0:
        raise ValueError("Неверная точность или лимит")
    history = [x0]
    for _ in range(max_iterations):
        x = history[-1]
        if df(x) == 0:
            raise ValueError("Нулевая производная")
        x_new = x - f(x) / df(x)
        if not isfinite(x_new):
            raise ValueError("Неконечное приближение")
        history.append(x_new)
        if abs(x_new - x) <= eps and abs(f(x_new)) <= eps:
            return history

    raise ValueError("Ньютон не сошёлся")


def root(a: float, b: float) -> float:
    if not isfinite(a) or not isfinite(b) or a >= b or f(a) * f(b) > 0:
        raise ValueError("Нужен отрезок со сменой знака")
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
    if len(argv) > 2:
        raise ValueError("Запуск: python task1.py [input.txt]")
    path = Path(argv[1]) if len(argv) == 2 else Path(__file__).parent / 'data/task1.txt'
    eps, a1, b1, a2, b2 = map(float, path.read_text(encoding='utf-8-sig').split())
    if not all(isfinite(v) for v in (eps, a1, b1, a2, b2)):
        raise ValueError("Нужны конечные данные")

    for a, b in ((a1, b1), (a2, b2)):
        x0_newton = start(a, b)
        lam = 1 / max((df(a), df(b)), key=abs)
        q = max(abs(1 - lam * df(x)) for x in (a, b))
        if not a <= phi(a, lam) <= phi(b, lam) <= b:
            raise ValueError("Итерации не сохраняют отрезок")
        print(f"[{a:g}; {b:g}]")
        report("Итерации", iteration((a + b) / 2, q, eps, lam))
        report("Ньютон", newton(x0_newton, eps))

if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, OverflowError) as error:
        raise SystemExit(str(error))
