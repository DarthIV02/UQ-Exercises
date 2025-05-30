import numpy as np
import numpy.typing as npt


def load_grades(filename: str) -> npt.NDArray:
    # TODO: read grades from the file.
    # ====================================================================

    with open(filename) as f:
        nums = [float(i) for i in f.readlines()]

    grades = np.array(nums, dtype=float)
    # ====================================================================
    return grades


def python_compute(array: npt.NDArray) -> tuple[float, float]:
    # TODO: compute the mean and the variance using standard Python.
    # ====================================================================
    mean = sum(array) / len(array)
    var = sum([(i - mean)**2 for i in array]) / (len(array)-1)
    # ====================================================================
    return mean, var


def numpy_compute(array: npt.NDArray, ddof: int = 0) -> tuple[float, float]:
    # TODO: compute the mean and the variance using numpy.
    # ====================================================================
    mean, var = np.mean(array), np.var(array, ddof=ddof)
    # ====================================================================
    return mean, var


if __name__ == "__main__":
    # TODO: load the grades from the file, compute the mean and the
    # variance using both implementations and report the results.
    # ====================================================================
    grades = load_grades('./bonus_exercise_1/template/data/G.txt')
    print("Grades: ", grades)
    
    mean_1, var_1 = python_compute(grades)
    mean_numpy, var_numpy = numpy_compute(grades, ddof=1)

    print("Mean in python: ", mean_1)
    print("Mean in numpy: ", mean_numpy)

    print("Var in python: ", var_1)
    print("Var in numpy: ", var_numpy)
    
    # ====================================================================
